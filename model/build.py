from .backbones.clip_model import CLIP2TextReID, convert_weights
import os.path as op
import torch.nn as nn
import numpy as np
import torch
from loss import *


class Pipeline(nn.Module):
    def __init__(self, backbone, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.embed_dim = args.embed_dim
        self.base = backbone
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.initial_T)) # 0.07
        self.criterion = self.get_criterion()

    
    def encode_image(self, image):
        image_features = self.base.encode_image(image)
        return image_features.float()

    def encode_text(self, text):
        text_features = self.base.encode_text(text)
        return text_features.float()
    
    def forward(self, image, caption, pid, image_id):
        image_features, text_features = self.base(image, caption)
        
        temperature = self.logit_scale.exp()
        loss = self.criterion(image_features.float(), text_features.float(), pid, image_id, temperature)

        return loss
    
    
    def get_criterion(self):
        criterion = {}
        if self.args.tcmpm == 'on':
            criterion['tcmpm'] = tcmpm_loss
        if self.args.cmpm == 'on':
            criterion['cmpm'] = cmpm_loss
        if self.args.ITC == 'on':
            criterion['ITC'] = ITC_loss

        def criterion_total(image_features, text_features, pid, image_id, temperature):
            loss = 0.
            if self.args.tcmpm == 'on':
                loss += criterion['tcmpm'](image_features, text_features, pid, image_id, temperature, label_mix=self.args.label_mix)
            if self.args.ITC == 'on':
                loss += criterion['ITC'](image_features, text_features, pid, temperature)
            if self.args.cmpm == 'on':
                loss += criterion['cmpm'](image_features, text_features, pid)
            return loss

        print("Loss functions: ", criterion.keys())
        return criterion_total



def build_model(args, num_classes=11003):

    backbone = CLIP2TextReID(embed_dim=args.embed_dim,                # 512
                          image_size=args.img_size,                   # (384, 128)
                          vision_layers=args.vision_layers,           # 12
                          vision_width=args.vision_width,             # 768
                          patch_size=args.patch_size,                 # 16
                          stride_size=args.stride_size,               # 16
                          context_length=args.text_length,            # 77
                          vocab_size=args.vocab_size,                 # 49408
                          transformer_width=args.transformer_width,   # 512
                          transformer_layers=args.transformer_layers, # 12
                          drop_rate=args.drop_rate,
                          drop_path_rate = args.drop_path_rate)
    convert_weights(backbone)
    # model.load_state_dict(state_dict)
    if args.training:
        if args.pretrain_type == 'CLIP-VIT-B-16':
            pretrained_path =op.join(args.root_dir, 'ViT-B-16.pt')
            backbone.load_param(pretrained_path)
            print(f"LOAD PRETRAINED MODEL FROM {pretrained_path}")
            
        elif args.pretrain_type == 'CLIP-VIT-L-14':
            pretrained_path =op.join(args.root_dir, 'ViT-L-14.pt')
            backbone.load_param(pretrained_path)
            
            print(f"LOAD PRETRAINED MODEL FROM {pretrained_path}")
        else:
            print(f"NO PRETRAINED PARAMETERS LOADED! TRAIN MODEL FROM SCRATCH.")
    
    return Pipeline(backbone, args, num_classes=num_classes)