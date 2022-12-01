import argparse


def get_args():
    parser = argparse.ArgumentParser(description="CLIP2TextReID")
    ###-----------------general settings----------------------###
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--output_dir", default="outputs")
    # parser.add_argument("--gpu_id", default="0", help="select gpu to run")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ###-----------------model general settings----------------------###
    parser.add_argument("--pretrain_type", default='CLIP-VIT-B-16') # whether use pretrained model
    parser.add_argument("--embed_dim", type=int, default=512, help="the final visual and textual feature dim")
    parser.add_argument("--drop_rate", type=float, default=0., help="dropout rate")
    parser.add_argument("--drop_path_rate", type=float, default=0., help="drop_path rate")
    parser.add_argument("--initial_T", type=float, default=0.07, help="initial temperature value")
    ###-----------------loss settings----------------------###
    parser.add_argument("--cmpm", default='off', help="whether use cmpm loss")
    parser.add_argument("--ITC", default='off', help="whether use image-text contrastive loss")
    parser.add_argument("--tcmpm", default='on', help="whether use tcmpm loss")
    ###-----------------vison trainsformer settings----------------------###
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--img_aug", default=False, action='store_true')
    parser.add_argument("--vision_layers", type=int, default=12)
    parser.add_argument("--vision_width", type=int, default=768)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--stride_size", type=int, default=16)
    ###-----------------text transformer settings----------------------###
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--transformer_layers", type=int, default=12)

    ### solver
    parser.add_argument("--learnable_loss_weight", default=False)
    parser.add_argument("--label_mix", default=False, action='store_true', help="whether mix pid and imagid label")
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    # scheduler
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="step")
    parser.add_argument("--target_lr", type=float, default=1e-5)
    parser.add_argument("--power", type=float, default=0.9)

    ### dataset
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="CUHK-PEDES or ICFG-PEDES")
    parser.add_argument("--sampler", default="random", help="choose sampler from type idtentity and random")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false') # whether in training mode

    args = parser.parse_args()

    return args