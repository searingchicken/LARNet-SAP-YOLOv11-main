import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
import warnings
from ultralytics import YOLO
import sys

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *

# Ignore warnings from YOLOv8
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='dehazing model name')
parser.add_argument('--num_workers', default=8, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()

# Dehaze model loading function
def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' prefix
        new_state_dict[name] = v
    return new_state_dict

# Test function for dehazing model
def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))  # Zhou Wang
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print(f'Test: [{idx}]\t'
              f'PSNR: {PSNR.val:.02f} ({PSNR.avg:.02f})\t'
              f'SSIM: {SSIM.val:.03f} ({SSIM.avg:.03f})')

        f_result.write(f'{filename},{psnr_val:.02f},{ssim_val:.03f}\n')

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    f_result.close()

    os.rename(os.path.join(result_dir, 'results.csv'),
              os.path.join(result_dir, f'{PSNR.avg:.02f} | {SSIM.avg:.04f}.csv'))

# YOLO inference function
def yolo_inference(dehazed_image_path, model, filename):
    # Perform object detection on the dehazed image
    result = model.predict(source=dehazed_image_path,
                           imgsz=640,
                           project='results/LARNet-SAP-YOLOv11',
                           name='test1',
                           save=True,
                           exist_ok=True)
    return result

if __name__ == '__main__':
    if args.model == 'LARNet':
        network = dehazeformer_b()  # Use dehazeformer_b for LARNet
    else:
        network = eval(args.model.replace('-', '_'))()
    network.cuda()

    saved_model_dir = os.path.join(args.save_dir, args.exp, f'{args.model}.pth')

    if os.path.exists(saved_model_dir):
        print(f'==> Start testing, current model name: {args.model}')
        network.load_state_dict(single(saved_model_dir))
    else:
        print('==> No existing trained model!')
        exit(0)

    try:
        yolo_model = YOLO('D:\\LARNet-SAP-YOLOv11-main\\runs\\SAP-YOLOv11\\train\\weights\\best.pt')
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        exit(1)

    # Setup dataset and dataloader for testing
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = PairLoader(dataset_dir, 'test', 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, args.dataset, args.model)
    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    torch.cuda.empty_cache()
    max_size = 1024
    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        filename = batch['filename'][0]

        _, _, H, W = input.size()
        orig_H, orig_W = H, W

        new_h = H if H % 2 == 0 else H + 1
        new_w = W if W % 2 == 0 else W + 1
        if new_h != H or new_w != W:
            input = F.interpolate(input, size=(new_h, new_w), mode='bilinear', align_corners=False)

        if max(new_h, new_w) > max_size:
            scale = max_size / max(new_h, new_w)
            resized_h, resized_w = int(new_h * scale), int(new_w * scale)
            input = F.interpolate(input, size=(resized_h, resized_w), mode='bilinear', align_corners=False)

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)

            if max(new_h, new_w) > max_size:
                output = F.interpolate(output, size=(orig_H, orig_W), mode='bilinear', align_corners=False)

            output = output * 0.5 + 0.5  # Normalize to [0, 1]

        dehazed_image_path = os.path.join(result_dir, 'imgs', f'{filename}')
        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(dehazed_image_path, out_img)

        print(f'Running YOLOv11 inference on {filename}')
        yolo_result = yolo_inference(dehazed_image_path, yolo_model, filename)

    print("Inference complete.")

