import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils
from model.utils.config import cfg
from model.rpn.rpn_fpn import _RPN_FPN
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

from ..darts import genotypes as gt
from ..darts.search_cell import SearchCell

from torch.autograd import Variable

import time
import pdb
    

class _Attention_DARTS(nn.Module):
    """ Attention_Darts """
    def __init__(self, classes, class_agnostic):
        super(_Attention_DARTS, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # darts 
        self._initialize_alphas()
        # attention
        self.in_channels = [256, 512, 1024, 2048]
        self.out_channels = 256
        self.num_ins = len(self.in_channels)
        self.num_outs = 5
        self.backbone_end_level = self.num_ins
        self.start_level = 0
        self.end_level = -1
        upsample_cfg = dict(mode='nearest')
        self.upsample_cfg = upsample_cfg.copy()
        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level) :
            l_conv = nn.Conv2d(self.in_channels[i], self.out_channels, kernel_size=1, stride=1, padding=0)
            fpn_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        no_channel = False
        no_spatial = False
        reduction_ratio = 16
        kernel_size = 7
        
        self.fusion_attentions = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            d_conv = nn.ModuleList()
            for j in range(self.start_level, self.backbone_end_level):
                temp = []
                for _ in range(i-j):
                    temp.append(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1))
                d_conv.append(nn.Sequential(*temp))
            self.fusion_attentions.append(SearchCell(
                                        gate_channels=self.out_channels * (self.num_ins - self.start_level),
                                        levels = self.num_ins - self.start_level, 
                                        reduction_ratio = reduction_ratio, 
                                        kernel_size= kernel_size
                                        ))
            self.downsample_convs.append(d_conv)
        
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.maxpool2d = nn.MaxPool2d(1, stride=2)
        # define rpn
        self.RCNN_rpn = _RPN_FPN(self.dout_base_model) #  self.dout_base_model = 256
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # NOTE: the original paper used pool_size = 7 for cls branch, and 14 for mask branch, to save the
        # computation time, we first use 14 as the pool_size, and then do stride=2 pooling for cls branch.
        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        # self.RCNN_roi_crop = _RoICrop()
    
    # darts
    def _initialize_alphas(self):
        n_ops = len(gt.PRIMITIVES)
        self.alphas = []
        for i in range(4):
            self.alphas.append (Variable(1e-3*torch.randn(n_ops), requires_grad=True))

    def genotype(self):
        gene_level0 = gt.parse(self.alphas[0])
        gene_level1 = gt.parse(self.alphas[1])
        gene_level2 = gt.parse(self.alphas[2])
        gene_level3 = gt.parse(self.alphas[3])
        
        return gt.Genotype(level0 = gene_level0, level1=gene_level1, level2=gene_level2, level3=gene_level3)

    def arch_parameters(self): 
        return self.alphas

    def weights(self):
        return self.net.parameters()
    
    def alphas(self):
        for n, p in self._alphas:
            yield p
    
    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # custom weights initialization called on netG and netD
        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.RCNN_toplayer, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        img_area = im_info[0][0] * im_info[0][1]
        h = rois.data[:, 4] - rois.data[:, 2] + 1
        w = rois.data[:, 3] - rois.data[:, 1] + 1
        roi_level = torch.log(torch.sqrt(h * w) / 224.0)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        roi_level.fill_(5)

        if cfg.POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                # feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l], scale)
                feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l])
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]

        elif cfg.POOLING_MODE == 'pool':
            roi_pool_feats = []
            box_to_levels = []
            for i, l in enumerate(range(2, 6)):
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze()
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                # feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l], scale)
                feat = self.RCNN_roi_pool(feat_maps[i], rois[idx_l])
                roi_pool_feats.append(feat)
            roi_pool_feat = torch.cat(roi_pool_feats, 0)
            box_to_level = torch.cat(box_to_levels, 0)
            idx_sorted, order = torch.sort(box_to_level)
            roi_pool_feat = roi_pool_feat[order]
            
        return roi_pool_feat

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # Bottom-up (backbone)
        c1 = self.RCNN_layer0(im_data) # c1 [1, 64, 150, 224]
        c2 = self.RCNN_layer1(c1) # c2 [1, 256, 150, 224]
        c3 = self.RCNN_layer2(c2) # c3 [1, 512, 75, 112]
        c4 = self.RCNN_layer3(c3) # c4 [1, 1024, 38, 56]
        c5 = self.RCNN_layer4(c4) # c5 [1, 2048, 19, 28]
        
        inputs = [c2, c3, c4 ,c5] # c2 : [N, 256, H, W], c3 : [N, 512, H, W], c4 : [N, 1024, H, W], c3 : [N, 2048, H, W]
        laterals = [
            lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Top-down
        weights = [F.softmax(alpha, dim=-1) for alpha in self.alphas]
        used_backbone_levels = len(laterals)
        outs = []
        for i in range(used_backbone_levels):
            shape = laterals[i].shape[2:]
            samples = []
            samples.extend([self.downsample_convs[i][j](laterals[j]) for j in range(i)])
            samples.append(laterals[i])
            samples.extend([F.interpolate(laterals[j], size=shape, **self.upsample_cfg) for j in range(i + 1, used_backbone_levels)])
            outs.append(self.fusion_attentions[i](samples, weights[i]))
        
        # len(outs) = 4 / outs0 [4, 256, 104, 168], outs1 [4, 256, 52, 84], outs2 [4, 256, 26, 42], outs3 [4, 256, 26, 42]
        
        p6 = self.maxpool2d(outs[3]) # P6 [N, 256, 256, 19]
        rpn_feature_maps = [outs[0], outs[1], outs[2], outs[3], p6]
        mrcnn_feature_maps = [outs[0], outs[1], outs[2], outs[3]]
        # rpn_feature_maps = [p2, p3, p4, p5, p6] 
        # mrcnn_feature_maps = [p2, p3, p4, p5]

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = rois_label.nonzero().squeeze()
            gt_assign_pos = gt_assign[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id

            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)
            rois_label = Variable(rois_label)

            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        # pooling features based on rois, output 14x14 map
        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(roi_pool_feat)


        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # loss (cross entropy) for object classification
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # loss (l1-norm) for bounding box regression
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        rois = rois.view(batch_size, -1, rois.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, gate_channels, levels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.gate_channels = gate_channels
        self.levels = levels
        self.mlp = nn.ModuleList()
        for _ in range(levels):
            if reduction_ratio != 1:
                self.mlp.append(nn.Sequential(
                    Flatten(),
                    nn.Linear(gate_channels, gate_channels // reduction_ratio),
                    nn.ReLU(),
                    nn.Linear(gate_channels // reduction_ratio, gate_channels // levels)
                ))
            else:
                self.mlp.append(nn.Sequential(
                    Flatten(),
                    nn.Linear(gate_channels, gate_channels // levels)
                ))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        attention = torch.cat(x, dim=1)
        scale = []
        for i in range(self.levels):
            for pool_type in self.pool_types:
                if pool_type == 'avg':
                    avg_pool = F.avg_pool2d(attention, (attention.size(2), attention.size(3)),
                                            stride=(attention.size(2), attention.size(3)))
                    channel_att_raw = self.mlp[i](avg_pool)
                elif pool_type == 'max':
                    max_pool = F.max_pool2d(attention, (attention.size(2), attention.size(3)),
                                            stride=(attention.size(2), attention.size(3)))
                    channel_att_raw = self.mlp[i](max_pool)
                elif pool_type == 'lp':
                    lp_pool = F.lp_pool2d(attention, 2, (attention.size(2), attention.size(3)),
                                          stride=(attention.size(2), attention.size(3)))
                    channel_att_raw = self.mlp[i](lp_pool)
                elif pool_type == 'lse':
                    # LSE pool only
                    lse_pool = logsumexp_2d(attention)
                    channel_att_raw = self.mlp[i](lse_pool)

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw
            scale.append(torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x[i]))
        return scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, levels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.levels = levels
        self.compress = ChannelPool()
        self.spatial = nn.ModuleList()
        for _ in range(levels):
            self.spatial.append(
                BasicConv(2 * levels, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False))

    def forward(self, x):
        attention = torch.cat([(self.compress(_)) for _ in x], dim=1)
        scale = [torch.sigmoid(self.spatial[i](attention)) for i in range(self.levels)]
        return scale

class FusionAttention(nn.Module):
    def __init__(self, gate_channels, levels, reduction_ratio=16, kernel_size=7, pool_types=['avg', 'max'],
                 no_channel=False, no_spatial=False):
        super(FusionAttention, self).__init__()
        self.levels = levels
        self.no_channel = no_channel
        self.no_spatial = no_spatial
        if not no_channel:
            self.channel_attention = ChannelAttention(gate_channels, levels, reduction_ratio, pool_types)
        if not no_spatial:
            self.spatial_attention = SpatialAttention(levels, kernel_size)

    def forward(self, x):
        if not self.no_channel:
            channel_attention_maps = self.channel_attention(x)
            x = [x[level] * channel_attention_maps[level] for level in range(self.levels)]

        if not self.no_spatial:
            spatial_attention_maps = self.spatial_attention(x)
            x = [x[level] * spatial_attention_maps[level] for level in range(self.levels)]
        return sum(x)