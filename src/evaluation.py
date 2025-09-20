from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool
from PIL import Image
import numpy as np
import shutil
import torch
import tqdm
import glob
import os
#import kornia 


            
class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}

def predict_tta(model, image, min_depth= 1e-3, max_depth = 80.):
    #model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pred = model(image)[-1].detach()
    #     pred = utils.depth_norm(pred)
    #     pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred = np.clip(pred.cpu().numpy(), 10, 1000)/100.
    pred = np.clip(pred.cpu().numpy(), min_depth, max_depth)

    image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(device)

    pred_lr = model(image)[-1].detach()
    #     pred_lr = utils.depth_norm(pred_lr)
    #     pred_lr = nn.functional.interpolate(pred_lr, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred_lr = np.clip(pred_lr.cpu().numpy()[...,::-1], 10, 1000)/100.
    pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], min_depth, max_depth)
    final = 0.5 * (pred + pred_lr)
    final = torch.nn.functional.interpolate(torch.Tensor(final), image.shape[-2:], mode='bilinear', align_corners=True)
    return torch.Tensor(final)


def eval(model, test_loader,  max_depth = 80. , min_depth = 1e-3 , gpus=None):
    if gpus is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = gpus[0]

    metrics = RunningAverageDict()
    # crop_size = (471 - 45, 601 - 41)
    # bins = utils.get_bins(100)
    total_invalid = 0
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in sequential:

            image = batch[0].to(device)
            gt = batch[-1].to(device)
            final = predict_tta(model, image)
            final = final.squeeze().cpu().numpy()

            # final[final < args.min_depth] = args.min_depth
            # final[final > args.max_depth] = args.max_depth
            final[np.isinf(final)] = max_depth
            final[np.isnan(final)] = min_depth

 
            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > min_depth , gt < max_depth)

            gt_height, gt_width = gt.shape
            eval_mask = np.zeros(valid_mask.shape)

            eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
            int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
    

            metrics.update(compute_errors(gt[valid_mask], final[valid_mask]))

    print(f"Total invalid: {total_invalid}")
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")
    return metrics


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    #err = np.log(pred) - np.log(gt)
    #silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, rmse_log=rmse_log, sq_rel=sq_rel)
