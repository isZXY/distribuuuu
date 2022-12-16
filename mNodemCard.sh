python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_net.py --cfg config/resnet18.yaml \
    OUT_DIR /tmp \
    MODEL.SYNCBN True \
    TRAIN.BATCH_SIZE 256
