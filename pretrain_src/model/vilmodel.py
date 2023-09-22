from ast import Add
import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad

from random import randint
from .zonemodel import AdaptiveZonePartition
from .zonemodel import gmap_embeds_attention_calculate
logger = logging.getLogger(__name__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        """
        hidden_states: (N, L_{hidden}, D)
        attention_mask: (N, H, L_{hidden}, L_{hidden})
        """
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                None if head_mask is None else head_mask[i],
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores

class GraphLXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):      
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]

        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

    def forward_lang2visn(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
    ):
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]
        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output

class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_l_layers = config.num_l_layers
        self.update_lang_bert = config.update_lang_bert

        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

    def forward(self, txt_embeds, txt_masks):
        extended_txt_masks = extend_neg_masks(txt_masks)
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]
        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()
        return txt_embeds
    
class CrossmodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks) # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds

class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(config.angle_feat_size + 3, config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.obj_linear = self.obj_layer_norm = None

        # 0: non-navigable, 1: navigable, 2: object
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None

    def forward(
        self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, type_embed_layer
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None

        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))

        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_fts))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      type_embed_layer(torch.ones(1, 1).long().to(device))
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens
        
class LocalVPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.encoder = CrossmodalEncoder(config)

    def vp_input_embedding(self, split_traj_embeds, split_traj_vp_lens, vp_pos_fts):
        vp_img_embeds = pad_tensors_wgrad([x[-1] for x in split_traj_embeds])
        vp_lens = torch.stack([x[-1]+1 for x in split_traj_vp_lens], 0)
        vp_masks = gen_seq_masks(vp_lens)
        max_vp_len = max(vp_lens)

        batch_size, _, hidden_size = vp_img_embeds.size()
        device = vp_img_embeds.device
        # add [stop] token at beginning
        vp_img_embeds = torch.cat(
            [torch.zeros(batch_size, 1, hidden_size).to(device), vp_img_embeds], 1
        )[:, :max_vp_len]
        vp_embeds = vp_img_embeds + self.vp_pos_embeddings(vp_pos_fts)

        return vp_embeds, vp_masks

    def forward(
        self, txt_embeds, txt_masks, split_traj_embeds, split_traj_vp_lens, vp_pos_fts
    ):
        vp_embeds, vp_masks = self.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_embeds = self.encoder(txt_embeds, txt_masks, vp_embeds, vp_masks)
        return vp_embeds


class ZoneGrader(nn.Module):
    def __init__(self, feature_dim, opts=None):
        super(ZoneGrader, self).__init__()
        self.opts = opts
        self.feature_dim = feature_dim
        self.out_feature_dim = self.opts.out_feature_dim
        self.txt_linear = nn.Linear(self.feature_dim, self.out_feature_dim)
        self.zone_linear = nn.Linear(self.feature_dim, self.out_feature_dim)
        self.txt_dropout = nn.Dropout(0.5)
        self.zone_dropout = nn.Dropout(0.5)
        self.sm = nn.Softmax(dim=1)

    def forward(self, txt_embeds, zone_embeds, mask=None, single=False):
        if single:
            txt_embeds = self.txt_linear(txt_embeds)
            txt_embeds = torch.tanh(txt_embeds)
            txt_embeds = self.txt_dropout(txt_embeds)
            zone_embeds = self.zone_linear(zone_embeds)
            zone_embeds = torch.tanh(zone_embeds)
            zone_embeds = self.zone_dropout(zone_embeds)
            attn = zone_embeds * txt_embeds
            attn = torch.sum(attn, dim = 1)
            attn /= math.sqrt(self.feature_dim)
            attn = torch.sigmoid(attn)      
        else:
            '''
            txt_embeds: batch x feature_dim
            zone_embeds: batch x zone_num x feature_dim
            '''
            txt_embeds = self.txt_linear(txt_embeds)
            txt_embeds = torch.tanh(txt_embeds)
            txt_embeds = self.txt_dropout(txt_embeds)
            zone_embeds = self.zone_linear(zone_embeds)
            zone_embeds = torch.tanh(zone_embeds)
            zone_embeds = self.zone_dropout(zone_embeds)
            txt_embeds = txt_embeds.unsqueeze(-1)         
            attn = torch.bmm(zone_embeds, txt_embeds)        
            attn = attn.squeeze(-1)             
            attn /= math.sqrt(self.feature_dim) 
            if mask is not None:
                attn = attn.masked_fill(mask, -float('inf'))
            if self.opts.zoner == "hard_zone":
                attn = self.sm(attn)
        return attn

class Zoner(nn.Module):
    def __init__(self, opts, config):
        super(Zoner, self).__init__()
        self.opts = opts
        self.config = config
        self.total_times = 0
        self.rezone_times = 0
        self.predict_rezone_times = 0
        # self.img_linear = nn.Linear(1, 1)
        self.zoner_size_for_random = self.opts.zoner_size_for_random
        self.feature_dim = self.config.hidden_size
        self.AZP = AdaptiveZonePartition(self.feature_dim, opts=self.opts)
        self.ZoneGrader = ZoneGrader(self.feature_dim, opts=self.opts)
        self.ZoneJudger = ZoneGrader(self.feature_dim, opts=self.opts)
        self.zoner_thres = self.opts.zoner_thres
        self.zoner_ratio = self.opts.zoner_ratio
        self.edge_thres = self.opts.edge_thres

    def zone_partition(self, gmap_lens, gmap_embeds, graph_sprels, zone_lens, rezone):
        batch_size = len(gmap_lens)
        point_size = gmap_embeds.size(1)
        gmap_vpzones = []
        gmap_ss = []
        zone_embeds = []
        zone_partition_logits = []
        if not self.opts.zoner_part_loss:
            zone_lens = []
        if self.opts.zoner == "random":
            for gl in gmap_lens:
                gmap_vpzones.append([0] + [randint(1, self.zoner_size_for_random) for i in range(gl-1)])
        elif self.opts.zoner == "hard_zone":
            for i in range(batch_size):
                N = gmap_lens[i]
                x = gmap_embeds[i, 1:N, :]
                my_graph_sprel = graph_sprels[i, 1:N, 1:N].clone()
                my_graph_sprel = my_graph_sprel < self.edge_thres
                my_graph_sprel = my_graph_sprel.to_sparse()
                edge_index = my_graph_sprel._indices()
                edge_weight = my_graph_sprel._values()
                if self.opts.zoner_part_loss:
                    gmap_vpzone, _, zone_embed, zone_partition_logit = self.AZP(x, edge_index, edge_weight, k=zone_lens[i])
                else:
                    gmap_vpzone, _, zone_embed, zone_partition_logit = self.AZP(x, edge_index, edge_weight)
                if self.opts.zoner_part_loss and rezone[i]:
                    zone_partition_logits.append(zone_partition_logit)
                if not self.opts.zoner_part_loss:
                    zone_lens.append(zone_embed.size(0))
                gmap_vpzones.append(gmap_vpzone)
                zone_embeds.append(zone_embed)
            if not self.opts.zoner_part_loss:
                zone_lens = torch.tensor(zone_lens).cuda()
            max_zone_len = max(zone_lens)
            if self.opts.zoner_part_loss:
                if rezone.any():
                    max_zone_len_for_partiotion = max(zone_lens[rezone])
                for i in range(len(zone_partition_logits)):
                    zone_partition_logits[i] = torch.cat([zone_partition_logits[i], torch.ones((zone_partition_logits[i].size(0) ,max_zone_len_for_partiotion - zone_partition_logits[i].size(1))).cuda() * -float("inf")], dim=1)
                if len(zone_partition_logits) != 0:
                    zone_partition_logits = torch.cat(zone_partition_logits, dim=0)
            for i in range(batch_size):
                zone_embeds[i] = torch.cat([zone_embeds[i], torch.zeros((max_zone_len - zone_lens[i], self.feature_dim)).cuda()], dim=0)
            zone_embeds = torch.stack(zone_embeds)
        elif self.opts.zoner == "soft_zone":
            for i in range(batch_size):
                N = gmap_lens[i]
                x = gmap_embeds[i, 0:N, :]
                my_graph_sprel = graph_sprels[i, 0:N, 0:N].clone()
                my_graph_sprel = my_graph_sprel < self.edge_thres
                my_graph_sprel = my_graph_sprel.to_sparse()
                edge_index = my_graph_sprel._indices()
                edge_weight = my_graph_sprel._values()
                if self.opts.zoner_part_loss:
                    _, S, zone_embed, zone_partition_logit = self.AZP(x, edge_index, edge_weight, k=zone_lens[i], point_size = point_size)
                else:
                    _, S, zone_embed, zone_partition_logit = self.AZP(x, edge_index, edge_weight, point_size = point_size)
                if self.opts.zoner_part_loss and rezone[i]:
                    zone_partition_logits.append(zone_partition_logit)
                if not self.opts.zoner_part_loss:
                    zone_lens.append(zone_embed.size(0))              
                gmap_ss.append(S)
                zone_embeds.append(zone_embed)
            if not self.opts.zoner_part_loss:
                zone_lens = torch.tensor(zone_lens).cuda()          
            max_zone_len = max(zone_lens)
            if self.opts.zoner_part_loss:
                if rezone.any():
                    max_zone_len_for_partiotion = max(zone_lens[rezone])
                for i in range(len(zone_partition_logits)):
                    zone_partition_logits[i] = torch.cat([zone_partition_logits[i], torch.ones((zone_partition_logits[i].size(0) ,max_zone_len_for_partiotion - zone_partition_logits[i].size(1))).cuda() * -float("inf")], dim=1)
                if len(zone_partition_logits)!=0:
                    zone_partition_logits = torch.cat(zone_partition_logits, dim=0)
            for i in range(batch_size):
                gmap_ss[i] = torch.cat([gmap_ss[i], torch.ones((point_size, max_zone_len - zone_lens[i])).cuda() * -float('inf')], dim=1)
                zone_embeds[i] = torch.cat([zone_embeds[i], torch.zeros((max_zone_len - zone_lens[i], self.feature_dim)).cuda()], dim=0)
            gmap_ss = torch.stack(gmap_ss)
            zone_embeds = torch.stack(zone_embeds)
        return gmap_vpzones, gmap_ss, zone_embeds, zone_lens, zone_partition_logits
    
    def zone_selection(self, txt_embeds, zone_embeds, txt_lens, zone_lens):
        if self.opts.zoner == "random":
            zone_logits = torch.rand((self.opts.batch_size, self.zoner_size_for_random+1)).cuda() # 0 is stop
        elif self.opts.zoner == "hard_zone" or self.opts.zoner == "soft_zone":
            txt_embeds = torch.mean(txt_embeds, dim=1)
            zone_masks = gen_seq_masks(zone_lens)
            zone_logits = self.ZoneGrader(txt_embeds, zone_embeds, mask=~zone_masks)
        return zone_logits

    def get_zone_label(self, global_act_labels, gmap_vpzones, gmap_ss, zone_logits):
        terminal = []
        for i in range(len(global_act_labels)):
            if global_act_labels[i] == -100:
                global_act_labels[i] = 0
                terminal.append(i)
        if self.opts.zoner == "random" or self.opts.zoner == "hard_zone":
            zone_label = torch.zeros_like(global_act_labels).cuda()
            for i in range(len(zone_label)):
                view_point_idx = global_act_labels[i]
                zone_label[i] = gmap_vpzones[i][view_point_idx]
        elif self.opts.zoner == "soft_zone":
            batch_size = len(gmap_ss)
            zone_label = gmap_ss[range(batch_size), global_act_labels, :]
            for t in terminal:
                zone_label[t] = zone_logits[t]
        return zone_label, zone_logits

    def update_zone(self, gmap_lens, global_act_labels, gmap_embeds, graph_sprels, txt_embeds, txt_lens, gmap_vpids, rezone, is_train, zone_lens, ori_zone_embeds):
        batch_size = len(gmap_lens)
        cur_zone_embeds = torch.zeros((batch_size, gmap_embeds.shape[2])).cuda()
        gmap_vpzones, gmap_ss, zone_embeds, zone_lens, zone_partition_logits = self.zone_partition(gmap_lens, gmap_embeds, graph_sprels, zone_lens, rezone)
        zone_logits = self.zone_selection(txt_embeds, zone_embeds, txt_lens, zone_lens)
        if global_act_labels != None:
            zone_label, zone_logits = self.get_zone_label(global_act_labels, gmap_vpzones, gmap_ss, zone_logits)
        else:
            zone_label = None
        predicted_zone = torch.argmax(zone_logits, 1)
        for i in range(batch_size):
            if rezone[i]:
                self.gmap_vpzones_dicts[i] = dict(zip(gmap_vpids[i], gmap_vpzones[i]))
                if is_train:
                    self.now_zone[i] = zone_label[i]
                else:
                    self.now_zone[i] = predicted_zone[i]
                cur_zone_embeds[i] = zone_embeds[i][self.now_zone[i].long()]
            else:
                for vpid in gmap_vpids[i]:
                    try:
                        self.gmap_vpzones_dicts[i][vpid]
                    except KeyError:
                        self.gmap_vpzones_dicts[i][vpid] = self.now_zone[i]
                if zone_label != None:
                    zone_label[i] = self.now_zone[i]
                predicted_zone[i] = self.now_zone[i]
                cur_zone_embeds[i] = ori_zone_embeds[i]
                gmap_vpzones[i] = []
                for vpid in gmap_vpids[i]:
                    gmap_vpzones[i].append(self.gmap_vpzones_dicts[i][vpid])
        return gmap_vpzones, gmap_ss, zone_logits, zone_label, predicted_zone, zone_partition_logits, cur_zone_embeds, zone_embeds
    
    def get_key_point(self, gmap_visited_vpids, key_point_len, gmap_step_ids):
        visited_len = len(gmap_visited_vpids)
        key_point = []
        gmap_step_ids_list = gmap_step_ids.tolist()
        step = (visited_len - 1) / key_point_len
        step = step.item()
        key = 1
        for i in range(key_point_len):
            if i == 0:
                key += step/2
            else:
                key += step
            try:
                key_point.append(gmap_step_ids_list.index(round(key)))
            except ValueError:
                temp_vpid = gmap_visited_vpids[round(key)]
                for j in range(visited_len-1,-1,-1):
                    if gmap_visited_vpids[j] == temp_vpid:
                        add_key = gmap_step_ids_list.index(j + 1)
                        key_point.append(add_key)
                        break

        return key_point
    def get_zone_partition_label(self, gmap_visited_vpids, gmap_step_ids, graph_sprels, gmap_lens, rezone):
        if not self.opts.zoner_part_loss:
            return None, None
        zone_lens = [math.ceil(len(i) * self.zoner_ratio) for i in gmap_visited_vpids]
        zone_lens = torch.tensor(zone_lens).cuda()
        batch_size = len(zone_lens)
        zone_partition_label = []
        for i in range(batch_size):
            if rezone[i]:
                N = gmap_lens[i]
                key_point = self.get_key_point(gmap_visited_vpids[i], zone_lens[i], gmap_step_ids[i])
                sprel = graph_sprels[i,key_point,1:N]
                zone_partition_label.append(torch.argmin(sprel, dim=0))
        if len(zone_partition_label) != 0:
            zone_partition_label = torch.cat(zone_partition_label)
        else:
            zone_partition_label = None
        return zone_partition_label, zone_lens

    def forward(self, gmap_lens, global_act_labels, gmap_embeds, graph_sprels, txt_embeds, txt_lens, now_pos, gmap_vpids, is_train, gmap_visited_vpids, gmap_step_ids):
        batch_size = len(gmap_lens)
        zone_judge_logits, zone_judge_labels = None, None
        if now_pos == None:
            now_pos = ['start'] * batch_size
        if self.opts.zoner == "soft_zone" or self.opts.zoner == "random":
            cur_zone_embeds = torch.zeros((batch_size, gmap_embeds.shape[2])).cuda()
            if now_pos[0] == 'start':
                self.now_zone = torch.zeros(batch_size).cuda()
            rezone = torch.tensor([True] * batch_size).cuda()
            for i in range(batch_size):
                if now_pos[i] == None:
                    rezone[i] = False
            zone_partition_label, zone_lens = self.get_zone_partition_label(gmap_visited_vpids, gmap_step_ids, graph_sprels, gmap_lens, rezone)
            gmap_vpzones, gmap_ss, zone_embeds, zone_lens, zone_partition_logits = self.zone_partition(gmap_lens, gmap_embeds, graph_sprels, zone_lens, rezone)
            zone_logits = self.zone_selection(txt_embeds, zone_embeds, txt_lens, zone_lens)
            predicted_zone = torch.argmax(zone_logits, 1)
            if global_act_labels != None:
                zone_label, zone_logits = self.get_zone_label(global_act_labels, gmap_vpzones, gmap_ss, zone_logits)
            else:
                zone_label = None
            for i in range(batch_size):
                if is_train:
                    if zone_label != None:
                        self.now_zone[i] = torch.argmax(zone_label[i], 0) # TODO
                    else:
                        self.now_zone[i] = predicted_zone[i] # few cases
                else:
                    self.now_zone[i] = predicted_zone[i]
                cur_zone_embeds[i] = zone_embeds[i][self.now_zone[i].long()]
            zone_embeds = cur_zone_embeds
        elif self.opts.zoner == "hard_zone":
            if now_pos[0] == 'start':
                self.total_times = self.total_times + batch_size
                self.rezone_times = self.rezone_times + batch_size
                self.predict_rezone_times = self.predict_rezone_times + batch_size
                self.gmap_vpzones_dicts = [{}] * batch_size
                self.now_zone = torch.zeros(batch_size).cuda()
                rezone = torch.tensor([True] * batch_size).cuda()
                zone_embeds = torch.zeros((batch_size, gmap_embeds.shape[2])).cuda()
                zone_partition_label, zone_lens = self.get_zone_partition_label(gmap_visited_vpids, gmap_step_ids, graph_sprels, gmap_lens, rezone)
                gmap_vpzones, gmap_ss, zone_logits, zone_label, predicted_zone, zone_partition_logits, zone_embeds, origin_zone_embeds = self.update_zone(gmap_lens, global_act_labels, gmap_embeds, graph_sprels, txt_embeds, txt_lens, gmap_vpids, rezone, is_train, zone_lens, zone_embeds)
                if zone_label != None:
                    predict_zone_embeds = origin_zone_embeds[range(batch_size), predicted_zone,:]
                    zone_scores = self.ZoneJudger(torch.mean(txt_embeds, dim=1), predict_zone_embeds, single=True)
                    zone_judge_logits = torch.stack([1-zone_scores, zone_scores], dim=1)    #0:rezone 1:not rezone
                    zone_judge_label = (predicted_zone == zone_label).long()  #False:rezone True:not rezone
                    
            else:
                now_pos_idnex = [gmap_vpids[i].index(now_pos[i]) for i in range(batch_size)]
                zone_embeds = gmap_embeds[range(batch_size), now_pos_idnex, :]
                zone_scores = self.ZoneJudger(torch.mean(txt_embeds, dim=1), zone_embeds, single=True)
                rezone = (zone_scores < self.zoner_thres)
                for i in range(batch_size):
                    if now_pos[i] == None:
                        rezone[i] = False
                self.predict_rezone_times = self.predict_rezone_times + sum(rezone==True)
                if is_train:
                    for i in range(batch_size):
                        if now_pos[i] == None:
                            continue
                        global_act_vpid = gmap_vpids[i][global_act_labels[i]]
                        if global_act_labels[i]!=0 and global_act_vpid in self.gmap_vpzones_dicts[i].keys() and self.gmap_vpzones_dicts[i][global_act_vpid] != self.now_zone[i]:
                            rezone[i] = True
                self.total_times = self.total_times + batch_size - now_pos.count(None)
                self.rezone_times = self.rezone_times + sum(rezone == True) 
                zone_partition_label, zone_lens = self.get_zone_partition_label(gmap_visited_vpids, gmap_step_ids, graph_sprels, gmap_lens, rezone)
                gmap_vpzones, gmap_ss, zone_logits, zone_label, predicted_zone, zone_partition_logits, zone_embeds, origin_zone_embeds = self.update_zone(gmap_lens, global_act_labels, gmap_embeds, graph_sprels, txt_embeds, txt_lens, gmap_vpids, rezone, is_train, zone_lens, zone_embeds)
        return gmap_vpzones, gmap_ss, zone_logits, zone_label, predicted_zone, rezone, zone_partition_label, zone_partition_logits, zone_embeds, zone_judge_logits, zone_judge_label
    
class GlobalMapEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.encoder = CrossmodalEncoder(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        )
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, global_act_labels=None, zoner=None, is_train=None, 
        graph_sprels=None, txt_embeds=None, txt_lens=None, opts=None
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)

        zone_logits, zone_label, gmap_embeds_attention, rezone, zone_partition_label, zone_partition_logits, zone_judge_logits, zone_judge_label = None, None, None, None, None, None, None, None

        if zoner == None:
            gmap_masks = gen_seq_masks(gmap_lens)
        elif zoner.opts.zoner == "hard_zone" or zoner.opts.zoner == "random":
            gmap_vp_zones, gmap_ss, zone_logits, zone_label, predicted_zone, rezone, zone_partition_label, zone_partition_logits, zone_embeds, zone_judge_logits, zone_judge_label = zoner(gmap_lens, global_act_labels, gmap_embeds, graph_sprels, txt_embeds, txt_lens, None, gmap_vpids, is_train, traj_vpids, gmap_step_ids)
            if is_train:
                if zone_label != None:
                    gmap_masks = gen_seq_masks(gmap_lens, zone_ops=True, gmap_vp_zones=gmap_vp_zones, zone_label=zone_label)
            else:
                gmap_masks = gen_seq_masks(gmap_lens, zone_ops=True, gmap_vp_zones=gmap_vp_zones, zone_label=predicted_zone)

        elif zoner.opts.zoner == "soft_zone":
            gmap_vp_zones, gmap_ss, zone_logits, zone_label, predicted_zone, rezone, zone_partition_label, zone_partition_logits, zone_embeds, zone_judge_logits, zone_judge_label = zoner(gmap_lens, global_act_labels, gmap_embeds, graph_sprels, txt_embeds, txt_lens, None, gmap_vpids, is_train, traj_vpids, gmap_step_ids)
            if is_train:
                if zone_label != None:
                    gmap_embeds_attention = gmap_embeds_attention_calculate(gmap_ss, zone_label, ~gmap_masks)
                else:
                    gmap_embeds_attention = None
            else:
                gmap_embeds_attention = gmap_embeds_attention_calculate(gmap_ss, zone_logits, ~gmap_masks)
            
        return gmap_embeds, gmap_masks, zone_logits, zone_label, gmap_embeds_attention, rezone, zone_partition_label, zone_partition_logits, zone_judge_logits, zone_judge_label
            
        
    def forward(
        self, txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, global_act_labels, graph_sprels=None, zoner=None, is_train=None, txt_lens=None, opts=None
    ):

        gmap_embeds, gmap_masks, zone_logits, zone_label, gmap_embeds_attention, rezone, zone_partition_label, zone_partition_logits, zone_judge_logits, zone_judge_label = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens, global_act_labels=global_act_labels, zoner=zoner, is_train=is_train, 
            graph_sprels=graph_sprels, txt_embeds=txt_embeds, txt_lens=txt_lens, opts=opts
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        gmap_embeds = self.encoder(
            txt_embeds, txt_masks, gmap_embeds, gmap_masks,
            graph_sprels=graph_sprels
        )
        return gmap_embeds, gmap_masks, zone_logits, zone_label, gmap_embeds_attention, rezone, zone_partition_label, zone_partition_logits, zone_judge_logits, zone_judge_label

# self.bert
class GlocalTextPathCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.lang_encoder = LanguageEncoder(config)

        self.img_embeddings = ImageEmbeddings(config)
        
        self.local_encoder = LocalVPEncoder(config)
        self.global_encoder = GlobalMapEncoder(config)
        self.config = config
        
        self.init_weights()

    def set_up(self, opts):
        self.opts = opts
        if self.opts.zoner != "null":
            self.zoner = Zoner(self.opts, self.config)

    def forward(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        return_gmap_embeds=True, global_act_labels=None
    ):        
        # text embedding
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        txt_masks = gen_seq_masks(txt_lens)
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)

        # trajectory embedding
        split_traj_embeds, split_traj_vp_lens = self.img_embeddings(
            traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens,
            self.embeddings.token_type_embeddings
        )
        
        # gmap embeds
        if return_gmap_embeds:
            gmap_embeds = self.global_encoder(
                txt_embeds, txt_masks,
                split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
                gmap_step_ids, gmap_pos_fts, gmap_lens, global_act_labels, graph_sprels=gmap_pair_dists
            )
        else:
            gmap_embeds = None

        # vp embeds
        vp_embeds = self.local_encoder(
            txt_embeds, txt_masks,
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )

        return gmap_embeds, vp_embeds

    def forward_mlm(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
    ):
        # text embedding
        txt_token_type_ids = torch.zeros_like(txt_ids)
        txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
        txt_masks = gen_seq_masks(txt_lens)
        txt_embeds = self.lang_encoder(txt_embeds, txt_masks)
        extended_txt_masks = extend_neg_masks(txt_masks)

        # trajectory embedding
        split_traj_embeds, split_traj_vp_lens = self.img_embeddings(
            traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens,
            self.embeddings.token_type_embeddings
        )
        
        # gmap embeds
        gmap_input_embeds, gmap_masks, _, _, _, _, _, _, _, _ = self.global_encoder.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )
        gmap_txt_embeds = txt_embeds
        extended_gmap_masks = extend_neg_masks(gmap_masks)
        for layer_module in self.global_encoder.encoder.x_layers:
            gmap_txt_embeds = layer_module.forward_lang2visn(
                gmap_txt_embeds, extended_txt_masks, 
                gmap_input_embeds, extended_gmap_masks,
            )

        # vp embeds
        vp_input_embeds, vp_masks = self.local_encoder.vp_input_embedding(
            split_traj_embeds, split_traj_vp_lens, vp_pos_fts
        )
        vp_txt_embeds = txt_embeds
        extended_vp_masks = extend_neg_masks(vp_masks)
        for layer_module in self.local_encoder.encoder.x_layers:
            vp_txt_embeds = layer_module.forward_lang2visn(
                vp_txt_embeds, extended_txt_masks, 
                vp_input_embeds, extended_vp_masks,
            )

        txt_embeds = gmap_txt_embeds + vp_txt_embeds
        return txt_embeds

    def forward_sap(
            self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=True, global_act_labels=None, is_train=None
        ):        
            # text embedding
            txt_token_type_ids = torch.zeros_like(txt_ids)
            txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
            txt_masks = gen_seq_masks(txt_lens)
            txt_embeds = self.lang_encoder(txt_embeds, txt_masks)

            # trajectory embedding
            split_traj_embeds, split_traj_vp_lens = self.img_embeddings(
                traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
                traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens,
                self.embeddings.token_type_embeddings
            )

            # gmap embeds
            if return_gmap_embeds:
                if self.opts.zoner != "null":
                    gmap_embeds, gmap_masks, zone_logits, zone_label, gmap_embeds_attention, rezone, zone_partition_label, zone_partition_logits, zone_judge_logits, zone_judge_label = self.global_encoder(
                        txt_embeds, txt_masks,
                        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids, gmap_step_ids, 
                        gmap_pos_fts, gmap_lens, global_act_labels, graph_sprels=gmap_pair_dists, zoner=self.zoner, is_train=is_train, txt_lens=txt_lens, opts=self.opts
                    )
                else:
                    gmap_embeds, gmap_masks, zone_logits, zone_label, gmap_embeds_attention, rezone, zone_partition_label, zone_partition_logits, zone_judge_logits, zone_judge_label = self.global_encoder(
                        txt_embeds, txt_masks,
                        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids, gmap_step_ids, 
                        gmap_pos_fts, gmap_lens, global_act_labels, graph_sprels=gmap_pair_dists, zoner=None, is_train=is_train, txt_lens=txt_lens, opts=self.opts
                    )
            else:
                gmap_embeds, gmap_masks, zone_logits, zone_label, gmap_embeds_attention, rezone, zone_partition_label, zone_partition_logits, zone_judge_logits, zone_judge_label = None, None, None, None, None, None, None, None, None, None

            # vp embeds
            vp_embeds = self.local_encoder(
                txt_embeds, txt_masks,
                split_traj_embeds, split_traj_vp_lens, vp_pos_fts
            )

            return gmap_embeds, vp_embeds, gmap_masks, zone_logits, zone_label, gmap_embeds_attention, rezone, zone_partition_label, zone_partition_logits, zone_judge_logits, zone_judge_label