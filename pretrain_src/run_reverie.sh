DATA_ROOT=datasets

NODE_RANK=0
NUM_GPUS=1
outdir=${DATA_ROOT}/REVERIE/exprs_map/pretrain/name

# cmt-clip_RN50-mlm.mrc.sap.og-init.lxmert-aug.speaker-HardZoner0.2-bs64
# step68000
# train
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    pretrain_src/train_reverie_obj.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config pretrain_src/config/reverie_obj_model_config.json \
    --config pretrain_src/config/reverie_obj_pretrain.json \
    --output_dir $outdir \
    --cache_dir ${DATA_ROOT}/huggingface/transformers/ \
    --data_root_dir ${DATA_ROOT}