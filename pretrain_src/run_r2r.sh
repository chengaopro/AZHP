DATA_ROOT=datasets

NODE_RANK=0
NUM_GPUS=1
outdir=${DATA_ROOT}/R2R/exprs_map/pretrain/name

# cmt-clip_RN50-mlm.mrc.sap-init.lxmert-aug.speaker-HardZoner
# step100000
# train
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --node_rank=${NODE_RANK} \
    pretrain_src/train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config pretrain_src/config/r2r_model_config.json \
    --config pretrain_src/config/r2r_pretrain.json \
    --output_dir $outdir \
    --cache_dir ${DATA_ROOT}/huggingface/transformers/ \
    --data_root_dir ${DATA_ROOT}
