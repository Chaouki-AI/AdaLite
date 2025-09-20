
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from datetime import datetime
from helper import load_models, load_images_deps, load_dl
from src.utils import RunningAverageDict, compute_errors


torch.cuda.empty_cache()

def predict_tta(args, model, image, device):
    pred = model(image)
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)

    image   = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(device)
    pred_lr = model(image)
    pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], args.min_depth, args.max_depth)

    final = 0.5 * (pred + pred_lr)
    final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:], mode='bilinear', align_corners=True)
    return torch.Tensor(final)

def eval(args, model, test_loader,  write=True):
    if args.gpu_ts :
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    max_depth = args.max_depth
    min_depth = args.min_depth
    metrics = RunningAverageDict()
    total_invalid = 0

    # Define column widths for formatting
    col_width = 12
    separator = "=" * (col_width * 9 + 10)  # Unified width for headers

    # Open file to store evaluation results only if write=True
    if write:
        file_path = f"./Evals_results/{args.backbone}_{args.dataset}_{datetime.now().strftime('%m-%d_%H-%M')}.txt"
        f = open(file_path, "w")
        f.write("Evaluation Metrics for Each Sample\n")
        f.write(separator + "\n")

        # Write header
        headers = ["Sample", "a1", "a2", "a3", "abs_rel", "sq_rel", "rmse", "log_10", "rmse_log", "silog"]
        f.write(" | ".join(h.ljust(col_width) for h in headers) + "\n")
        f.write(separator + "\n")

    with torch.no_grad():
        model.eval()
        sequential = test_loader
        for batch_idx, batch in enumerate(tqdm(sequential)):
            image = batch['image'].to(device)
            gt = batch['depth'].to(device)
            

            final = predict_tta(args, model, image, device)
            
            final = final.squeeze().cpu().numpy()

            final[np.isinf(final)] = max_depth
            final[np.isnan(final)] = min_depth

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    total_invalid += 1
                    continue

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > min_depth, gt <= max_depth)  #gt < max_depth

            # Apply cropping if necessary
            gt_height, gt_width = gt.shape
            eval_mask = np.ones(valid_mask.shape) if args.dataset == 'sunrgbd' else np.zeros(valid_mask.shape)

            if args.crop == "garg":
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.crop == "eigen":
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
            batch_metrics = compute_errors(gt[valid_mask], final[valid_mask])
            metrics.update(batch_metrics)

            # Write formatted line for this sample if write=True
            if write:
                f.write(f"{str(batch_idx+1).ljust(col_width)} | " + 
                        " | ".join(f"{batch_metrics[k]:.4f}".ljust(col_width) for k in headers[1:]) + "\n")

    # Print and store final aggregated metrics
    metrics = {k: round(v, 4) for k, v in metrics.get_value().items()}

    if write:
        # Write final results
        f.write(separator + "\n")
        f.write("Final Averaged Metrics:\n")
        f.write(separator + "\n")
        f.write(f"{'Overall'.ljust(col_width)} | " + 
                " | ".join(f"{metrics[k]:.4f}".ljust(col_width) for k in headers[1:]) + "\n")
        f.write(separator + "\n")
        f.close()
        print(f"Metrics saved to {file_path}")

    print(f"Metrics: {metrics}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    
    # Model's Specifications
    parser.add_argument("--backbone", default='b1', choices=['b1', 'b0', 'joint', 'adabins', 'fastdepth'],
                         help='backbone model to be used in the model') 
    parser.add_argument("--super", action="store_true", 
                        help="Choose whether using a Gan or only Encoder-Decoder model")  
    parser.add_argument("--basic", action="store_true", 
                        help="Choose whether using a Gan or only Encoder-Decoder model")  
    parser.add_argument("--path-pth-model", default=None,
                        help='pretrained pth file that be use for init intialize for the model')
    # Dataset parameters 
    parser.add_argument("--dataset", default="kitti", type=str, 
                        help='dataset used for training, kitti or nyu')
    parser.add_argument("--test-txt", default="./Data/splits/kitti/kitti_eigen_test_files_with_gt.txt", type=str, 
                        help='path to the filenames text file for testing')
    parser.add_argument("--images-path", default="./Data/images/", type=str, 
                        help='path of images')
    parser.add_argument("--depths-path", default="./Data/depths/", type=str, 
                        help='path of depths')
    parser.add_argument("--max_depth", type=float, default=10., 
                        help='maximum depth in estimation')
    parser.add_argument("--min_depth", type=float, default=1e-3,
                        help='minimum depth in estimation')
    parser.add_argument("--crop", default="eigen", type=str, help="type of the Crop to be performed for evaluation",
                        choices=['eigen', 'garg'])
    
    #GPU choice
    parser.add_argument("--gpu-ts", default=False, action="store_true",
                        help="Use the GPU or No for testing and evaluation")

    args = parser.parse_args()

    model = load_models(args)[1] if args.backbone =='adabins' else load_models(args)[0]
    imgs_ts, deps_ts = load_images_deps(args, train=False)
    valid_dl = load_dl(args, imgs=imgs_ts[:], depths=deps_ts[:], train=False)
    #criterion_ueff = SILogLoss(args)

    eval(args, model, valid_dl,  write=True)

 