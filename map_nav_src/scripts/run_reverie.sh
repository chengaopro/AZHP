DATA_ROOT=datasets

### training ###
train_alg=iterative

features=clip_RN50
ft_dim=1024
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

# iterative-clip_RN50-seed.0-clip_RN50-HardZoner0.2-bs64-step68000-hardZoneKey0.2.0.5-pretrained-rlWeight0.2-critic0.05-bert+clip_RN50-reproduce
name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-clip_RN50-HardZoner0.2-bs64-step68000
name=${name}-hardZoneKey0.2.0.5-pretrained-rlWeight0.2-critic0.05-bert+clip_RN50-reproduce

outdir=${DATA_ROOT}/REVERIE/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --cache_dir ${DATA_ROOT}/huggingface/transformers/
      --dataset reverie
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
      
      --max_action_len 15
      --max_instr_len 200
      --max_objects 20

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
      --rl_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0.

      --zoner hard_zone
      --zoner_is_pretrained
      --zoner_part_loss
      --keep_zone_partition_loss
      --zoner_ratio 0.2
      --zoner_thres 0.5
      --out_feature_dim 128
      --edge_thres 5.0
      "

CUDA_VISIBLE_DEVICES='0' python map_nav_src/reverie/main_nav_obj.py $flag  \
      --bert_ckpt_file ${DATA_ROOT}/REVERIE/exprs_map/pretrain/cmt-clip_RN50-mlm.mrc.sap.og-init.lxmert-aug.speaker-HardZoner0.2-bs64/ckpts/model_step_68000.pt \
      --eval_first






