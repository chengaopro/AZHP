from ast import arg
import torch


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    if args.cache_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    

    return tokenizer

def get_vlnbert_models(args, config=None):
    
    from transformers import PretrainedConfig
    from models.vilmodel import GlocalTextPathNavCMT
    
    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path)
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                k = k[7:]    
            if '_head' in k or 'sap_fuse' in k:
                new_ckpt_weights['bert.' + k] = v
            else:
                new_ckpt_weights[k] = v
            
    if args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    if args.cache_dir:
        vis_config = PretrainedConfig.from_pretrained(args.cache_dir)
    else:
        vis_config = PretrainedConfig.from_pretrained(cfg_name)
    

    if args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2
    
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_pano_layers = args.num_pano_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.graph_sprels = args.graph_sprels
    vis_config.glocal_fuse = args.fusion == 'dynamic'

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_pano_embedding = args.fix_pano_embedding
    vis_config.fix_local_branch = args.fix_local_branch

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.attention_probs_dropout_prob = 0.1
    vis_config.hidden_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False
    vis_config.zoner_size_for_random = args.zoner_size_for_random
    vis_config.zoner = args.zoner
    vis_config.zoner_ratio = args.zoner_ratio
    vis_config.zoner_thres = args.zoner_thres
    vis_config.batch_size = args.batch_size
    vis_config.zoner_is_pretrained = args.zoner_is_pretrained
    vis_config.zoner_part_loss = args.zoner_part_loss
    vis_config.out_feature_dim = args.out_feature_dim
    vis_config.edge_thres = args.edge_thres
    vis_config.keep_zone_partition_loss = args.keep_zone_partition_loss

    vis_config.test = args.test
    vis_config.vis = args.vis
    vis_config.abla_mode = args.abla_mode
    
    vis_config.tokenizer = args.tokenizer
    if 'clip' in args.tokenizer:
        vis_config.clip_model_path = args.clip_model_path
        vis_config.lang_feat_size = args.lang_feat_size
        
    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)

    if not args.zoner_is_pretrained:
        visual_model.set_up()
        
    return visual_model