# node 2:
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.19" \
    --master_port=29500 \
    train_net.py --cfg config/resnet18.yaml