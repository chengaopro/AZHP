import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r', 'r4r'])
    parser.add_argument('--output_dir', type=str, default='default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default='')

    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    parser.add_argument('--act_visited_nodes', action='store_true', default=False)
    parser.add_argument('--fusion', choices=['global', 'local', 'avg', 'dynamic'])
    parser.add_argument('--expl_sample', action='store_true', default=False)
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--expert_policy', default='spl', choices=['spl', 'ndtw'])

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=100000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--eval_first', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=80)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument("--rl_weight", type=float, default=0.50)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)

    parser.add_argument("--features", type=str, default='vitbase')

    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_pano_embedding', action='store_true', default=False)
    parser.add_argument('--fix_local_branch', action='store_true', default=False)

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_x_layers', type=int, default=4)

    parser.add_argument('--enc_full_graph', default=False, action='store_true')
    parser.add_argument('--graph_sprels', action='store_true', default=False)

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.3)

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_backtrack', action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='rms',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=2048)
    parser.add_argument('--obj_feat_size', type=int, default=0)
    parser.add_argument('--views', type=int, default=36)

    # Zoner
    parser.add_argument("--zoner", type=str, default="hard_zone")
    parser.add_argument('--zoner_is_pretrained', action='store_true', default=False)
    parser.add_argument("--zoner_size_for_random", type=int, default=2)
    parser.add_argument("--zoner_ratio", type=float, default=0.2)
    parser.add_argument("--zoner_thres", type=float, default=0.3)
    parser.add_argument('--zoner_part_loss', action='store_true', default=False)
    parser.add_argument('--keep_zone_partition_loss', action='store_true', default=False)
    parser.add_argument("--out_feature_dim", type=int, default=32)
    parser.add_argument("--edge_thres", type=float, default=5.0)
    
    parser.add_argument("--abla_mode", type=int, default=4)
    parser.add_argument('--vis', action='store_true', default=False)
    
    # A2C
    parser.add_argument("--gamma", default=0.9, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg', 
        choices=['imitation', 'dagger', 'iterative'], 
        default='imitation'
    )

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    ft_file_map = {
        'vitbase': 'pth_vit_base_patch16_224_imagenet.hdf5',
        'clip_ViTB': 'CLIP-ViT-B-32-views.hdf5',
        'clip_RN50': 'CLIP-ResNet-50-views.hdf5',
    }
    if 'clip' in args.tokenizer:
        clip_model_map = {
            'clip_RN50': 'RN50.pt',
            'clip_ViTB': 'ViT-B-32.pt', 
            'bert+clip_RN50': 'RN50.pt',
            'bert+clip_ViTB': 'ViT-B-32.pt'
        }
        lang_feat_size_map = {
            'clip_RN50': 1024,
            'clip_ViTB': 512, 
            'bert+clip_RN50': 1024,
            'bert+clip_ViTB': 512, 
        }
        args.clip_model_path = os.path.join(ROOTDIR, 'clip_models', clip_model_map[args.tokenizer])
        args.lang_feat_size = lang_feat_size_map[args.tokenizer]
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args

