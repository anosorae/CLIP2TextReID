from prettytable import PrettyTable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn.parallel
import numpy as np
import time
import os.path as op

# import clip
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    parser.add_argument("--config_file", default='./outputs/best/logs/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.dataset_name = 'ICFG-PEDES'

    log_dir = args.log_dir

    args.training = False
    logger = setup_logger('CLIP2TextReID.text', save_dir=log_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    # load clip pretrained model
    # model, preprocess = clip.load("ViT-B/16", device=device)

    # get image text dataloader
    # test_img_loader, test_txt_loader = build_dataloader(args, preprocess)
    test_img_loader, test_txt_loader = build_dataloader(args)
    model = build_model(args)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    cmc, mAP = evaluator.eval(model.eval())
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP"])
    table.float_format = '.4'
    table.add_row(['t2i', cmc[0], cmc[4], cmc[9], mAP])
    logger.info("Validation Results: ")
    logger.info('\n' + str(table))