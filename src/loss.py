import torch
import warnings
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Tuple, Union
from pytorch3d.loss import chamfer_distance


class KLLoss(nn.Module): 
    def __init__(self, args):
        super(KLLoss, self).__init__()
        self.name = 'KLLoss'
        self.args = args
    def forward(self, target, input, mask=None, interpolate=True):
        #mask = target>=1e-3
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)    
        if mask is not None:
            input  = input[mask]  + 1e-8
            target = target[mask] + 1e-8
        #result = torch.sum(-target * input.log(), dim=1).mean()
        T = 2.0  # temperature


        # Teacher probs (soft targets)
        P = F.softmax(target / T, dim=1)        # (B, C, H, W)

        # Student log-probs
        Q_log = F.log_softmax(input / T, dim=1) # (B, C, H, W)

        # KL divergence (pixel-wise)
        result = F.kl_div(Q_log, P, reduction="batchmean") * (T * T)


        return result #self.loss(input, target).mean()

class DistilationLoss(nn.Module): 
    def __init__(self, args):
        super(DistilationLoss, self).__init__()
        self.name = 'DistilationLoss'
        self.args = args
        self.distance = nn.KLDivLoss(reduction="batchmean")
    def forward(self, model, adabins):
        loss = 0 
        feat_model, feat_adabins = [i for i in model.decoder.features], [i for i in adabins.features]
        

        for i, j in zip(feat_model, feat_adabins):
            i = nn.functional.interpolate(i, j.shape[-2:], mode='bilinear', align_corners=True) 
            loss += self.kl_loss(i, j) * 0.5
        return 0.001 * loss 
    
    def prepare(self, input):
        target_points = input.flatten(1)
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(feats2.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)
        return target_points, target_lengths

    def loss(self, feats1, feats2):
        feats1, len1 = self.prepare(feats1)
        feats2, len2 = self.prepare(feats2)
        loss, _ = chamfer_distance(x=feats1, y=feats2, y_lengths=len2)
        return loss.mean()
        
    def kl_loss(self, pr, gt):
        pr = F.softmax(pr, dim=1).flatten(1)
        gt = F.softmax(gt, dim=1).flatten(1)
        pr = pr / pr.sum(1, keepdim = True)
        gt = gt / gt.sum(1, keepdim = True)
        return self.distance(pr, gt)


class SILogLoss(nn.Module): 
    def __init__(self, args):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.args = args
    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)    
        if mask is not None:
            input  = input[mask]  + 1e-8
            target = target[mask] + 1e-8
        g = torch.log(input) - torch.log(target)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)


class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss,self).__init__()
        self.mse = nn.MSELoss()
        self.grad_factor = 10.
        self.normal_factor = 1.
        self.criterion = 'l1gn'  # l1, l2, rmsle + gn

       
    def forward(self,pred,target):
        if 'l1' in self.criterion:
            depth_loss = self.L1_imp_Loss(pred,target)
        elif 'l2' in self.criterion:
            depth_loss = self.L2_imp_Loss(pred,target)
        elif 'rmsle' in self.criterion:
            depth_loss = self.RMSLELoss(pred,target)
        if 'gn' in self.criterion:
            grad_target, grad_pred = self.imgrad_yx(target), self.imgrad_yx(pred)
            grad_loss = self.GradLoss(grad_pred, grad_target)     * self.grad_factor
            normal_loss = self.NormLoss(grad_pred, grad_target) * self.normal_factor
            return depth_loss + grad_loss + normal_loss
        else:
            return depth_loss
    
    def GradLoss(self,grad_target,grad_pred):
        return torch.sum( torch.mean( torch.abs(grad_target-grad_pred) ) )
    
    def NormLoss(self, grad_target, grad_pred):
        prod = ( grad_pred[:,:,None,:] @ grad_target[:,:,:,None] ).squeeze(-1).squeeze(-1)
        pred_norm = torch.sqrt( torch.sum( grad_pred**2, dim=-1 ) )
        target_norm = torch.sqrt( torch.sum( grad_target**2, dim=-1 ) ) 
        return 1 - torch.mean( prod/(pred_norm*target_norm) )
    
    def RMSLELoss(self, pred, target):
        return torch.sqrt(self.mse(torch.log(pred + 0.5), torch.log(target + 0.5)))
 
        
    
    def L1_imp_Loss(self, pred, target):
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
    
    def L2_imp_Loss(self, pred, target):
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss
    
    def imgrad_yx(self,img):
        N,C,_,_ = img.size()
        grad_y, grad_x = self.imgrad(img)
        return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)
    
    def imgrad(self,img):
        img = torch.mean(img, 1, True)
        fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv1.weight = nn.Parameter(weight)
        grad_x = conv1(img)

        fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv2.weight = nn.Parameter(weight)
        grad_y = conv2(img)
        return grad_y, grad_x
    

class mixelLoss(nn.Module):
    def __init__(self, args):
        super(mixelLoss, self).__init__()
        self.args = args
        self.silog = SILogLoss(args)
        self.depth = DepthLoss()
        
    def forward(self, input, target, mask=None, interpolate=True):
        loss_silog = self.silog(input, target, mask, interpolate) * 0.9
        loss_depth = self.depth(input, target) * 0.1
        return loss_silog  + loss_depth 
    