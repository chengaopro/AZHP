import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from reverie.agent_obj import GMapObjectNavAgent
from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad
from torch.autograd import Variable


class SoonGMapObjectNavAgent(GMapObjectNavAgent):

    def get_results(self):
        output = [{'instr_id': k, 
                    'trajectory': {
                        'path': v['path'], 
                        'obj_heading': [v['pred_obj_direction'][0]],
                        'obj_elevation': [v['pred_obj_direction'][1]],
                    }} for k, v in self.results.items()]
        return output

    def rollout(self, train_ml=None, train_rl=False, reset=True, is_train=None):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'pred_obj_direction': None,
            'details': {},
        } for ob in obs]

        # Language input: txt_ids, txt_masks
        if 'clip' in self.args.tokenizer:
            language_inputs = self._language_variable(obs, 'clip')
            txt_embeds = self.vln_bert('language_clip', language_inputs)
            if 'bert' in self.args.tokenizer:
                language_inputs = self._language_variable(obs, 'bert')
                txt_embeds_bert = self.vln_bert('language', language_inputs)
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
                        'og_direction': obs[i]['obj_directions'][torch.argmax(i_obj_logits)] if len(i_objids) > 0 else None,
                        'og_details': {'objids': i_objids, 'logits': i_obj_logits[:len(i_objids)]},
                    }
                                        
            if train_ml is not None:
                ml_loss += self.criterion(nav_logits, nav_targets)
                if self.args.fusion in ['avg', 'dynamic'] and self.args.loss_nav_3:
                    # add global and local losses
                    ml_loss += self.criterion(nav_outs['global_logits'], nav_targets)    # global
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
                if self.args.keep_zone_partition_loss and self.args.zoner in ["soft_zone", "hard_zone"] and zone_partition_label != None:
                    zone_partition_loss = F.cross_entropy(zone_partition_logits, zone_partition_label, reduction='sum')
                    if batch_size_valid != 0:
                        zone_partition_loss = zone_partition_loss * 0.1 / batch_size_valid
                        self.loss = self.loss + zone_partition_loss
                        self.logs['Zone_partition_loss'].append(zone_partition_loss.item())
                if self.args.zoner == "hard_zone":
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
                    traj[i]['pred_obj_direction'] = stop_score['og_direction']
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
              