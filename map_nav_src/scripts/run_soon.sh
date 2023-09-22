DATA_ROOT=datasets

### training ###
train_alg=iterative

features=clip_RN50
ft_dim=1024
obj_features=butd
obj_ft_dim=2048

ngpus=1
seed=0

# iterative-clip_RN50-seed.0-clip_RN50-bs16-step18k-hardZoneKey0.2.0.5-rlWeight0.02-critic0.05-bert-bs8
name=${train_alg}-${features}
name=${name}-seed.${seed} 
name=${name}-clip_RN50-bs16-step18k
name=${name}-hardZoneKey0.2.0.5-rlWeight0.02-critic0.05-bert-bs8

outdir=${DATA_ROOT}/SOON/exprs_map/finetune/${name}


flag="--root_dir ${DATA_ROOT}
      --cache_dir ${DATA_ROOT}/huggingface/transformers/
      --dataset soon
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 20
      --max_instr_len 100
      --max_objects 100

      --batch_size 8
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2   
      --rl_weight 0.02

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0.

      --zoner hard_zone
      --zoner_part_loss
      --zoner_size_for_random 2
      --zoner_ratio 0.2
      --zoner_thres 0.5
      --out_feature_dim 128
      --edge_thres 5.0
      "


CUDA_VISIBLE_DEVICES='0' python map_nav_src/soon/main.py $flag  \
      --bert_ckpt_file ${DATA_ROOT}/SOON/exprs_map/pretrain/cmt-clip_RN50.butdobj-mlm.sap.og-init.lxmert-noZoner-bs16/ckpts/model_step_18000.pt \
      --eval_first
