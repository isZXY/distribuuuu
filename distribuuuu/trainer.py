import os
import time

import timm
import torch
import torch.nn as nn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from distribuuuu import models, utils
from distribuuuu.config import cfg


def train_epoch(train_loader, net, criterion, optimizer, cur_epoch, start_epoch, tic):
    """Train one epoch"""
    time_start = time.time()
    rank = torch.distributed.get_rank()  # 获得进程序号
    time_end = time.time()
    process_time = time_end - time_start
    logger.info(" 获得进程序号耗时 {} s".format(process_time))

    time_start = time.time()
    batch_time, data_time, losses, top1, topk = utils.construct_meters()
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, topk],
        prefix=f"TRAIN:  [{cur_epoch + 1}]",
    )
    time_end = time.time()
    process_time = time_end - time_start
    logger.info("设置测量耗时 {} s".format(process_time))
    # Set learning rate
    lr = utils.get_epoch_lr(cur_epoch)
    utils.set_lr(optimizer, lr)
    if rank == 0:
        logger.debug(
            f"CURRENT EPOCH: {cur_epoch + 1:3d},   LR: {lr:.4f},   POLICY: {cfg.OPTIM.LR_POLICY}"
        )

    # Set sampler
    train_loader.sampler.set_epoch(cur_epoch)

    acc_train_time = 0
    acc_reduce_time = 0
    acc_update_time = 0
    acc_updatelog_time = 0
    # time_end = time.time()
    # train_time = time_end - time_start
    # logger.info(" 获得进程序号耗时 {} s".format(acc_process_time))

    net.train()
    end = time.time()
    for idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)
        time_start = time.time()
        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)

        acc_1, acc_k = utils.accuracy(outputs, targets, topk=(1, cfg.TRAIN.TOPK))
        time_end = time.time()
        train_time = time_end - time_start
        acc_train_time += train_time

        time_start = time.time()
        loss, acc_1, acc_k = utils.scaled_all_reduce([loss, acc_1, acc_k])
        time_end = time.time()
        reduce_time = time_end - time_start
        acc_reduce_time += reduce_time

        time_start = time.time()
        losses.update(loss.item(), batch_size)
        top1.update(acc_1[0].item(), batch_size)
        topk.update(acc_k[0].item(), batch_size)
        time_end = time.time()
        acc_updatelog_time += (time_end - time_start)

        time_start = time.time()
        batch_time.update(time.time() - end)
        time_end = time.time()
        update_time = time_end - time_start
        acc_update_time += update_time

        end = time.time()

        if rank == 0 and (
                (idx + 1) % cfg.TRAIN.PRINT_FREQ == 0 or (idx + 1) == len(train_loader)
        ):
            progress.cal_eta(idx + 1, len(train_loader), tic, cur_epoch, start_epoch)
            progress.display(idx + 1)

    return acc_train_time, acc_reduce_time, acc_update_time, acc_updatelog_time


def validate(val_loader, net, criterion):
    """Validte the model"""
    rank = torch.distributed.get_rank()
    batch_time, data_time, losses, top1, topk = utils.construct_meters()
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1, topk],
        prefix="VAL:  ",
    )

    net.eval()
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets) in enumerate(val_loader):
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            acc_1, acc_k = utils.accuracy(outputs, targets, topk=(1, cfg.TRAIN.TOPK))
            loss, acc_1, acc_k = utils.scaled_all_reduce([loss, acc_1, acc_k])

            batch_size = inputs.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc_1[0].item(), batch_size)
            topk.update(acc_k[0].item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if rank == 0 and (
                    (idx + 1) % cfg.TEST.PRINT_FREQ == 0 or (idx + 1) == len(val_loader)
            ):
                progress.display(idx + 1)

    return top1.avg, topk.avg


def train_model():
    """Train a model"""

    # Set up distributed device
    time_start = time.time()
    utils.setup_distributed()  # DDP初始化设置,使用torch._C._distributed_c10d 设置DDP backend(NCCL)

    rank = int(os.environ["RANK"])  # 确定该机器为matser or worker
    local_rank = int(os.environ["LOCAL_RANK"])  # 进程号
    device = torch.device("cuda", local_rank)  # 确定device

    utils.setup_seed(rank)
    utils.setup_logger(rank, local_rank)
    try:
        net = models.build_model(  # 建立模型
            arch=cfg.MODEL.ARCH,
            pretrained=cfg.MODEL.PRETRAINED,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )
    except KeyError:
        net = timm.create_model(
            model_name=cfg.MODEL.ARCH,
            pretrained=cfg.MODEL.PRETRAINED,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )
    time_end = time.time()
    logger.info(" setup_distributed 耗时 {} s".format(time_end - time_start))

    # SyncBN (https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html)
    time_start = time.time()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net) if cfg.MODEL.SYNCBN else net
    net = net.to(device)
    # DistributedDataParallel Wrapper
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)  # DDP Wrapper
    time_end = time.time()

    logger.info(" 设置并推送模型耗时 {} s".format(time_end - time_start))

    time_start = time.time()
    train_loader = utils.construct_train_loader()
    val_loader = utils.construct_val_loader()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = utils.construct_optimizer(net)
    time_end = time.time()

    logger.info(" (本地)加载数据集和优化器耗时 {} s".format(time_end - time_start))

    # Resume from a specific checkpoint or the last checkpoint
    best_acc1 = start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and utils.has_checkpoint():
        file = utils.get_last_checkpoint()
        start_epoch, best_acc1 = utils.load_checkpoint(file, net, optimizer)
    elif cfg.MODEL.WEIGHTS:
        load_opt = optimizer if cfg.TRAIN.LOAD_OPT else None
        start_epoch, best_acc1 = utils.load_checkpoint(cfg.MODEL.WEIGHTS, net, load_opt)

    if rank == 0:
        # from torch.utils.collect_env import get_pretty_env_info
        # logger.debug(get_pretty_env_info())
        # logger.debug(net)
        logger.info("\n\n\n            =======  TRAINING  ======= \n\n")
        logger.info(utils.count_parameters(net))

    tic = time.time()
    period_allreduce = 0
    period_local_train = 0
    period_local_update = 0
    period_vaildate = 0
    period_log_update = 0
    for epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train one epoch
        acc_train_time, acc_reduce_time, acc_update_time, acc_updatelog_time = train_epoch(train_loader, net, criterion,
                                                                                           optimizer, epoch,
                                                                                           start_epoch, tic)

        logger.info("epoch {} train 耗时 {} s".format(epoch, acc_train_time))
        logger.info("epoch {} reduce 耗时 {} s".format(epoch, acc_reduce_time))
        logger.info("epoch {} update 耗时 {} s".format(epoch, acc_update_time))
        logger.info("epoch {} update_log 耗时 {} s".format(epoch, acc_updatelog_time))

        period_local_train += acc_train_time
        period_allreduce += acc_reduce_time
        period_local_update += acc_update_time
        period_log_update += acc_updatelog_time

        # Validate
        start = time.time()
        acc1, acck = validate(val_loader, net, criterion)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        # Save model
        checkpoint_file = utils.save_checkpoint(
            net, optimizer, epoch, best_acc1, is_best
        )
        if rank == 0:
            logger.info(
                f"ACCURACY: TOP1 {acc1:.3f}(BEST {best_acc1:.3f}) | TOP{cfg.TRAIN.TOPK} {acck:.3f} | SAVED {checkpoint_file}"
            )
        end = time.time()
        logger.info("epoch {} 验证耗时 {} s".format(epoch, end - start))
        period_vaildate += (end - start)

    logger.info("**!**总train耗时 {} s".format(period_local_train))
    logger.info("**!**总update耗时 {} s".format(period_local_update))
    logger.info("**!**总allreduce耗时 {} s".format(period_allreduce))
    logger.info("**!**总period_log_update耗时 {} s".format(period_log_update))
    logger.info("**!**总验证耗时 {} s".format(period_vaildate))


def test_model():
    """Test a model"""

    utils.setup_distributed()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    utils.setup_logger(rank, local_rank)

    try:
        net = models.build_model(
            arch=cfg.MODEL.ARCH,
            pretrained=cfg.MODEL.PRETRAINED,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )
    except KeyError:
        net = timm.create_model(
            model_name=cfg.MODEL.ARCH,
            pretrained=cfg.MODEL.PRETRAINED,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )
    net = net.to(device)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    val_loader = utils.construct_val_loader()
    criterion = nn.CrossEntropyLoss().to(device)

    if cfg.MODEL.WEIGHTS:
        utils.load_checkpoint(cfg.MODEL.WEIGHTS, net)

    acc1, acck = validate(val_loader, net, criterion)
    if rank == 0:
        logger.info(f"ACCURACY: TOP1 {acc1:.3f}  |  TOP{cfg.TRAIN.TOPK} {acck:.3f}")
