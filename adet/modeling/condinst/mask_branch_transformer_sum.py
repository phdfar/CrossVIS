import math
from typing import Dict

import torch
from detectron2.layers import ShapeSpec
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn

from adet.layers import conv_with_kaiming_uniform
from adet.utils.comm import aligned_bilinear

INF = 100000000


def build_mask_branch(cfg, input_shape):
    return MaskBranch(cfg, input_shape)


class SpatialTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=3):
        super(SpatialTransformerEncoder, self).__init__()

        # Define the positional encoding for the spatial dimensions
        h, w = 48, 80  # Height and width of the input images
        pos_enc_h = nn.Parameter(torch.randn(h, d_model // 2))
        pos_enc_w = nn.Parameter(torch.randn(w, d_model // 2))

        # Define the TransformerEncoder layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_layers)

        # Save the parameters
        self.pos_enc_h = pos_enc_h
        self.pos_enc_w = pos_enc_w

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Compute the positional encodings for the spatial dimensions
        pos_emb_h = self.pos_enc_h.unsqueeze(1).repeat(1, width, 1)  # Shape: (height, width, d_model // 2)
        pos_emb_w = self.pos_enc_w.unsqueeze(0).repeat(height, 1, 1)  # Shape: (height, width, d_model // 2)
        pos_emb = torch.cat([pos_emb_h, pos_emb_w], dim=-1)  # Shape: (height, width, d_model)

        # Add the positional encoding to the input tensor
        x = x + pos_emb.permute(2, 0, 1).unsqueeze(0)  # Shape: (batch_size, channels, height, width, d_model)

        # Reshape the input tensor for processing by the TransformerEncoder
        x = x.permute(0, 3, 2, 1).reshape(batch_size * width, height, channels)

        # Apply the TransformerEncoder layer to each spatial location separately
        x = self.transformer_encoder(x)

        # Reshape the output tensor back to the original shape
        x = x.reshape(batch_size, width, height, channels, -1).permute(0, 3, 2, 1, 4)
        
        return x.squeeze()

class MaskBranch(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES
        self.sem_loss_on = cfg.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON
        self.num_outputs = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        norm = cfg.MODEL.CONDINST.MASK_BRANCH.NORM
        num_convs = cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS
        channels = cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS
        self.out_stride = input_shape[self.in_features[0]].stride
        
        device = torch.device('cuda')

        #channels = channels//2

        #channels = 256
        self.transformer_layer = SpatialTransformerEncoder(d_model=channels, nhead=8, num_layers=3).to(device)


        feature_channels = {k: v.channels for k, v in input_shape.items()}

        conv_block = conv_with_kaiming_uniform(norm, activation=True)

        self.conv_blockz = conv_block(256 , channels, 3, 1)

        self.refine = nn.ModuleList()
        for in_feature in self.in_features:
            self.refine.append(
                conv_block(feature_channels[in_feature], channels, 3, 1))


        self.final_conv=nn.Conv2d(channels, max(self.num_outputs, 1), 1)
        
        tower = []
        for i in range(num_convs):
            tower.append(conv_block(channels, channels, 3, 1))
        #tower.append(nn.Conv2d(channels, max(self.num_outputs, 1), 1))
        self.add_module('tower', nn.Sequential(*tower))

        if self.sem_loss_on:
            num_classes = cfg.MODEL.FCOS.NUM_CLASSES
            self.focal_loss_alpha = cfg.MODEL.FCOS.LOSS_ALPHA
            self.focal_loss_gamma = cfg.MODEL.FCOS.LOSS_GAMMA

            in_channels = feature_channels[self.in_features[0]]
            self.seg_head = nn.Sequential(
                conv_block(in_channels, channels, kernel_size=3, stride=1),
                conv_block(channels, channels, kernel_size=3, stride=1))

            self.logits = nn.Conv2d(channels,
                                    num_classes,
                                    kernel_size=1,
                                    stride=1)

            prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)

    def forward(self, features, gt_instances=None):
        
        """
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.refine[i](features[f])
            else:
                x_p = self.refine[i](features[f])

                target_h, target_w = x.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                x = x + x_p
        """
        input_tensor = self.conv_blockz(features['p3'])
        
        batch_size, num_channels, height, width = input_tensor.shape

        if height != 48 or width != 80:
            input_tensor = torch.nn.functional.interpolate(input_tensor, size=(48, 80), mode='bilinear', align_corners=False)
            x = self.transformer_layer(input_tensor)
            try:
              x = torch.nn.functional.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
            except:
              x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False)

        else:
            #print('input_tensor',input_tensor.size())
            x = self.transformer_layer(input_tensor)
            #print('x',x.size())

        try:
          x = self.tower(x) + x
          mask_feats = self.final_conv(x)
        except:
          x = x.unsqueeze(0)
          x = self.tower(x) + x
          mask_feats = self.final_conv(x)


        if self.num_outputs == 0:
            mask_feats = mask_feats[:, :self.num_outputs]

        losses = {}
        # auxiliary thing semantic loss
        if self.training and self.sem_loss_on:
            logits_pred = self.logits(
                self.seg_head(features[self.in_features[0]]))

            # compute semantic targets
            semantic_targets = []
            for per_im_gt in gt_instances:
                h, w = per_im_gt.gt_bitmasks_full.size()[-2:]
                areas = per_im_gt.gt_bitmasks_full.sum(dim=-1).sum(dim=-1)
                areas = areas[:, None, None].repeat(1, h, w)
                areas[per_im_gt.gt_bitmasks_full == 0] = INF
                areas = areas.permute(1, 2, 0).reshape(h * w, -1)
                min_areas, inds = areas.min(dim=1)
                per_im_sematic_targets = per_im_gt.gt_classes[inds] + 1
                per_im_sematic_targets[min_areas == INF] = 0
                per_im_sematic_targets = per_im_sematic_targets.reshape(h, w)
                semantic_targets.append(per_im_sematic_targets)

            semantic_targets = torch.stack(semantic_targets, dim=0)

            # resize target to reduce memory
            semantic_targets = semantic_targets[:, None, self.out_stride //
                                                2::self.out_stride,
                                                self.out_stride //
                                                2::self.out_stride]

            # prepare one-hot targets
            num_classes = logits_pred.size(1)
            class_range = torch.arange(num_classes,
                                       dtype=logits_pred.dtype,
                                       device=logits_pred.device)[:, None,
                                                                  None]
            class_range = class_range + 1
            one_hot = (semantic_targets == class_range).float()
            num_pos = (one_hot > 0).sum().float().clamp(min=1.0)

            loss_sem = sigmoid_focal_loss_jit(
                logits_pred,
                one_hot,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction='sum',
            ) / num_pos
            losses['loss_sem'] = loss_sem

        return mask_feats, losses
