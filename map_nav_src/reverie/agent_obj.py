import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
# import line_profiler

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent

from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad
from torch.autograd import Variable
from models.vis import vis_record, vis_scans, vis_instr_ids, vis_time_step


class GMapObjectNavAgent(Seq2SeqAgent):
    
    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        self.critic_high = Critic(self.args).cuda() 
        # buffer
        self.scanvp_cands = {}

    def _language_variable(self, obs, tokenizer):
        if 'clip' in tokenizer:
            seq_lengths = [ob['instr_clip_length'] for ob in obs]
            seq_tensor = np.zeros((len(obs), 77), dtype=np.int64)
            mask = np.zeros((len(obs), 77), dtype=np.bool)
        else:
            seq_lengths = [len(ob['instr_encoding']) for ob in obs]
            seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
            mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool)
        for i, ob in enumerate(obs):
            if 'clip' in tokenizer:
                seq_tensor[i] = ob['instr_clip_encoding']
            else:
                seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_obj_img_fts, batch_loc_fts, batch_nav_types = [], [], [], []
        batch_view_lens, batch_obj_lens = [], []
        batch_cand_vpids, batch_objids = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            # object
            obj_loc_fts = np.concatenate([ob['obj_ang_fts'], ob['obj_box_fts']], 1)
            nav_types.extend([2] * len(obj_loc_fts))
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_obj_img_fts.append(torch.from_numpy(ob['obj_img_fts']))
            batch_loc_fts.append(torch.from_numpy(np.concatenate([view_loc_fts, obj_loc_fts], 0)))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_objids.append(ob['obj_ids'])
            batch_view_lens.append(len(view_img_fts))
            batch_obj_lens.append(len(ob['obj_img_fts']))

        # pad features to max_len
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_obj_img_fts = pad_tensors(batch_obj_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()
        batch_obj_lens = torch.LongTensor(batch_obj_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'obj_img_fts': batch_obj_img_fts, 
            'loc_fts': batch_loc_fts, 'nav_types': batch_nav_types,
            'view_lens': batch_view_lens, 'obj_lens': batch_obj_lens,
            'cand_vpids': batch_cand_vpids, 'obj_ids': batch_objids,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        batch_gmap_visited_vpids = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if gmap.graph.visited(k):
                    visited_vpids.append(k)
                else:
                    unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))
            batch_gmap_visited_vpids.append(visited_vpids)

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left, 'gmap_visited_vpids': batch_gmap_visited_vpids, 
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, obj_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i], 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)
        vp_obj_masks = torch.cat([torch.zeros(batch_size, 1).bool().cuda(), nav_types == 2], 1)

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+obj_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_obj_masks': vp_obj_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    def _teacher_object(self, obs, ended, view_lens):
        targets = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:
                targets[i] = self.args.ignoreid
            else:
                i_vp = ob['viewpoint']
                if i_vp not in ob['gt_end_vps']:
                    targets[i] = self.args.ignoreid
                else:
                    if ob['gt_obj_id'] is not None:
                        i_objids = ob['obj_ids']
                        targets[i] = self.args.ignoreid
                        for j, obj_id in enumerate(i_objids):
                            if str(obj_id) == str(ob['gt_obj_id']):
                                targets[i] = j + view_lens[i] + 1
                                break
                    else:
                        targets[i] = self.args.ignoreid
        return torch.from_numpy(targets).cuda()

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True, is_train=None):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)

        if self.args.test and self.args.vis:
            global vis_scans
            global vis_instr_ids
            vis_scans.clear()
            vis_instr_ids.clear()
            for i in range(batch_size):
                vis_scans += [obs[i]['scan']]
                vis_instr_ids += [obs[i]['instr_id']]
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'pred_objid': None,
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        if 'clip' in self.args.tokenizer:
            language_inputs = self._language_variable(obs, 'clip')
            txt_embeds = self.vln_bert('language_clip', language_inputs) # output embedding
            if 'bert' in self.args.tokenizer:
                language_inputs = self._language_variable(obs, 'bert')
                txt_embeds_bert = self.vln_bert('language', language_inputs) # output embedding
                if txt_embeds_bert.shape[1] > txt_embeds.shape[1]:
                    txt_embeds_bert[:, :txt_embeds.shape[1], :] = txt_embeds_bert[:, :txt_embeds.shape[1], :] + txt_embeds
                    txt_embeds = txt_embeds_bert
                else:
                    txt_embeds = txt_embeds_bert + txt_embeds[:, :txt_embeds_bert.shape[1], :]
        else:
            language_inputs = self._language_variable(obs, 'bert')
            txt_embeds = self.vln_bert('language', language_inputs)

        if train_rl:
            # The init distance from the view point to the target
            last_dist = np.zeros(batch_size, np.float32)
            for i, ob in enumerate(obs):
                last_dist[i] = ob['distance']
        
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.     
        og_loss = 0.   
        if train_rl:
            rewards = []
            hidden_states = []
            hidden_states_H = []
            policy_log_probs = []
            high_policy_log_probs = []
        self.now_pos = ["start"] * batch_size

        self.vln_bert.vln_bert.zoner.total_times = 0
        self.vln_bert.vln_bert.zoner.rezone_times = 0
        self.vln_bert.vln_bert.zoner.predict_rezone_times = 0

        for t in range(self.args.max_action_len):
            if self.args.test and self.args.vis:
                global vis_time_step
                vis_time_step[0] = t
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'], 
                    pano_inputs['view_lens'], pano_inputs['obj_lens'], 
                    pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
                'is_train': is_train,
                'now_pos': self.now_pos
            })

            if self.args.fusion == 'local':
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_vpids = nav_inputs['gmap_vpids']

            if train_ml is not None:
                # Supervised training
                nav_targets = self._teacher_action(
                    obs, nav_vpids, ended, 
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None
                )

                nav_inputs.update({
                    'global_act_labels': nav_targets.clone(),
                })
            else:                
                nav_inputs.update({
                    'global_act_labels': None,
                })

            nav_outs = self.vln_bert('navigation', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
            else:
                nav_logits = nav_outs['fused_logits']
                if train_rl:
                    nav_states = nav_outs['hidden_states']
                    hidden_states.append(nav_states.detach())
                    hidden_states_H.append(nav_outs['cur_zone_states'].detach()) 
            
            nav_probs = torch.softmax(nav_logits, 1)
            obj_logits = nav_outs['obj_logits']
           
            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    # update i_vp: stop and object grounding scores
                    i_objids = obs[i]['obj_ids']
                    i_obj_logits = obj_logits[i, pano_inputs['view_lens'][i]+1:]
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                        'og': i_objids[torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                        'og_details': {'objids': i_objids, 'logits': i_obj_logits[:len(i_objids)]},
                    }

            if train_ml is not None:
                ml_loss += self.criterion(nav_logits, nav_targets) # fuse loss
                if self.args.fusion in ['avg', 'dynamic'] and self.args.loss_nav_3:
                    # add global and local losses
                    # ml_loss += self.criterion(nav_outs['global_logits'], nav_targets)    # global
                    local_nav_targets = self._teacher_action(
                        obs, nav_inputs['vp_cand_vpids'], ended, visited_masks=None
                    )
                    ml_loss += self.criterion(nav_outs['local_logits'], local_nav_targets)  # local
                # objec grounding 
                obj_targets = self._teacher_object(obs, ended, pano_inputs['view_lens'])
                og_loss += self.criterion(obj_logits, obj_targets)
                zone_logits = nav_outs['zone_logits']
                zone_label = nav_outs['zone_label']
                rezone = nav_outs['rezone']
                zone_partition_label = nav_outs['zone_partition_label']
                zone_partition_logits = nav_outs['zone_partition_logits']
                if self.args.zoner in ["soft_zone", "hard_zone"]:
                    batch_size_valid = sum(rezone)
                if self.args.abla_mode > 1 and self.args.keep_zone_partition_loss and self.args.zoner in ["soft_zone", "hard_zone"] and zone_partition_label != None:
                    zone_partition_loss = F.cross_entropy(zone_partition_logits, zone_partition_label, reduction='sum')
                    if batch_size_valid != 0:
                        zone_partition_loss = zone_partition_loss * 0.1 / batch_size_valid
                        self.loss = self.loss + zone_partition_loss
                        self.logs['Zone_partition_loss'].append(zone_partition_loss.item())
                if self.args.abla_mode > 2 and self.args.zoner == "hard_zone":
                    zone_loss = F.cross_entropy(zone_logits, zone_label, reduction='none')
                    zone_loss = zone_loss * rezone
                    zone_loss = torch.sum(zone_loss)
                    if batch_size_valid != 0:
                        zone_loss = zone_loss * 2.5 / batch_size_valid
                    self.loss = self.loss + zone_loss
                    self.logs['Zone_selection_loss'].append(zone_loss.item())
                elif self.args.zoner == "soft_zone":
                    zone_logits = F.log_softmax(zone_logits, dim=1)
                    zone_logits = zone_logits.masked_fill(zone_logits==-float("inf"), 0)
                    zone_label = F.log_softmax(zone_label, dim=1)
                    zone_label = zone_label.masked_fill(zone_label==-float("inf"), 0)
                    zone_loss = F.kl_div(zone_logits, zone_label, reduction='sum', log_target=True)
                    if batch_size_valid != 0:
                        zone_loss = zone_loss * 500 / batch_size_valid
                    self.loss = self.loss + zone_loss
                    self.logs['Zone_selection_loss'].append(zone_loss.item())     

            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                if train_rl:
                    policy_log_probs.append(c.log_prob(a_t))
                    high_policy_log_probs.append(nav_outs['high_policy_log_probs'])
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])
            self.now_pos = cpu_a_t       

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf'), 'og': None}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    traj[i]['pred_objid'] = stop_score['og']
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                                'obj_ids': [str(x) for x in v['og_details']['objids']],
                                'obj_logits': v['og_details']['logits'].tolist(),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)
                    
                    
            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        if action_idx == None:
                            if dist[i] < 3.0:
                                reward[i] = 10
                            else:
                                reward[i] = -10
                        else:
                            reward[i] = - (dist[i] - last_dist[i])
                        # Miss the target penalty
                        if (last_dist[i] <= 3.0) and (dist[i] > 3.0):
                            reward[i] = reward[i] - 10
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                
            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))
            # Early exit if all ended
            if ended.all():
                break
    
        if train_ml is not None:
            rezone_ratio = self.vln_bert.vln_bert.zoner.rezone_times / max(self.vln_bert.vln_bert.zoner.total_times, 1)
            predict_rezone_ratio = self.vln_bert.vln_bert.zoner.predict_rezone_times / max(self.vln_bert.vln_bert.zoner.total_times, 1)
            if self.args.zoner == 'hard_zone':
                self.logs['rezone_ratio'].append(rezone_ratio.item())
                self.logs['predict_rezone_ratio'].append(predict_rezone_ratio.item())
            else:
                self.logs['rezone_ratio'].append(rezone_ratio)
                self.logs['predict_rezone_ratio'].append(predict_rezone_ratio)
        
        if train_rl:
            # t+1 step in A2C # TODO adjust this part for reverie
            # update gmaps according to current obs
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1
            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'], 
                    pano_inputs['view_lens'], pano_inputs['obj_lens'], 
                    pano_inputs['nav_types'],
                )
            )
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
                'is_train': is_train,
                'now_pos': self.now_pos
            })
            nav_outs = self.vln_bert('navigation', nav_inputs)
            last_hidden_states = nav_outs['hidden_states']
            last_hidden_states_H = nav_outs['cur_zone_states']
        
            
            rl_loss = 0.0
            
            last_value_L = self.critic(last_hidden_states).detach() 
            discount_reward = np.zeros(batch_size, np.float32)        
            for i in range(batch_size):
                if not ended[i]:        
                    discount_reward[i] = last_value_L[i]
            length = len(rewards)
            total = 0
            rewards_H = [0] * length
            for t in range(length-1, -1, -1):
                tmp = 0.0
                for tt in range(t):
                    tmp = tmp + rewards[tt]
                rewards_H[t] = tmp
                
                discount_reward = rewards[t] + discount_reward * 0.9  
                clip_reward = discount_reward.copy()
                td_target = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                state_value = self.critic(hidden_states[t].detach()) 
                td_error = (state_value - td_target).detach()  
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                
                rl_loss += (td_error * policy_log_probs[t] * mask_).sum() 
                rl_loss += (((state_value - td_target) ** 2) * mask_).sum() * 0.05 
                self.logs['low_critic_loss'].append((((state_value - td_target) ** 2) * mask_).sum().item() * 0.05)
                rl_loss += (- 0.01 * entropys[t] * mask_).sum()  
                total = total + np.sum(masks[t])
            self.logs['total'].append(total)
            rl_loss = rl_loss * train_rl / batch_size
            self.loss += rl_loss
            self.logs['low_RL_loss'].append(rl_loss.item())

            
            high_rl_loss = 0.0
            
            last_value_H = self.critic_high(last_hidden_states_H).detach()
            discount_reward_H = np.zeros(batch_size, np.float32) 
            for i in range(batch_size):
                if not ended[i]:
                    discount_reward_H = last_value_H[i]       
            for t in range(length-1, -1, -1):
                discount_reward_H = rewards_H[t] + discount_reward_H * 0.9  
                clip_reward_H = discount_reward_H.copy()
                td_target_H = Variable(torch.from_numpy(clip_reward_H), requires_grad=False).cuda()
                state_value_H = self.critic_high(hidden_states_H[t].detach()) 
                td_error_H = (state_value_H - td_target_H).detach()
                
                high_rl_loss += (td_error_H * high_policy_log_probs[t] * mask_).sum()  
                high_rl_loss += (((state_value_H - td_target) ** 2) * mask_).sum() * 0.05   
                self.logs['high_critic_loss'].append((((state_value_H - td_target) ** 2) * mask_).sum().item() * 0.05)
            high_rl_loss = high_rl_loss * train_rl / batch_size
            self.high_rl_loss = self.high_rl_loss + high_rl_loss
            self.logs['high_RL_loss'].append(high_rl_loss.item())

        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            og_loss = og_loss * train_ml / batch_size
            self.loss += ml_loss
            self.loss += og_loss
            self.logs['IL_loss'].append(ml_loss.item())
            self.logs['OG_loss'].append(og_loss.item())

        return traj
