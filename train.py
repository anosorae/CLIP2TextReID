import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import logging
from prettytable import PrettyTable
import torch
import numpy as np
import random
import time
import os.path as op

from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from options import get_args
from utils.comm import get_rank, synchronize
from utils.meter import AverageMeter


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0 

    logger = logging.getLogger("CLIP2TextReID.train")
    logger.info('start training')

    loss_meter = AverageMeter()
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        loss_meter.reset()
        model.train()
        for n_iter, (pid, image_id,  img, caption) in enumerate(train_loader):
            pid = pid.to(device)
            image_id = image_id.to(device)
            img = img.to(device)
            caption = caption.to(device)

            loss = model(img, caption, pid, image_id)
            loss_meter.update(loss.item(), img.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, scheduler.get_lr()[0]))


        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', 1 / model.logit_scale.exp(), epoch)
        tb_writer.add_scalar('loss', loss_meter.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                if args.distributed:
                    cmc, mAP = evaluator.eval(model.module.eval())
                cmc, mAP = evaluator.eval(model.eval())
                table = PrettyTable(["task", "R1", "R5", "R10", "mAP"])
                table.float_format = '.4'
                table.add_row(['t2i', cmc[0], cmc[4], cmc[9], mAP])
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info('\n' + str(table))
                torch.cuda.empty_cache()
                if best_top1 < cmc[0]:
                    best_top1 = cmc[0]
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join('experiments/', args.dataset_name, args.output_dir, f'{cur_time}_{name}')
    args.log_dir = op.join(args.output_dir, args.log_dir)
    logger = setup_logger('CLIP2TextReID', save_dir=args.log_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.log_dir, args)

    

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    model.to(device)
    

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)