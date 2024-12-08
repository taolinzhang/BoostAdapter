import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *
from loguru import logger
import sys
from IPython import embed
from utils import AverageMeter
import time
import copy

def get_plpd_image(x):
    import torchvision
    from einops import rearrange
    patch_len = 4
    resize_t = torchvision.transforms.Resize(((x.shape[-1]//patch_len)*patch_len,(x.shape[-1]//patch_len)*patch_len))
    resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
    x = resize_t(x)
    x = rearrange(x, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
    perm_idx = torch.argsort(torch.rand(x.shape[0],x.shape[1]), dim=-1)
    x = x[torch.arange(x.shape[0]).unsqueeze(-1),perm_idx]
    x = rearrange(x, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)
    x = resize_o(x)
    return x

def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='../data/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--delta', type=int, default=0)
    parser.add_argument('--views', type=int, default=64)

    args = parser.parse_args()

    return args


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False, fifo=True):
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            cache[pred].append(item)
            if fifo: 
                if len(cache[pred])>shot_capacity:
                    cache[pred] = cache[pred][1:]
            else:
                cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
                cache[pred] = cache[pred][:shot_capacity]   
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)
        if len(cache_keys) == 0:
            return torch.zeros(1)[0]

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # return alpha * cache_logits, affinity
        return alpha * cache_logits

def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, logger, args, cfg):
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    batch_time = AverageMeter('Time', ':6.3f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(loader),
        [top1],
        prefix='Test: ')
    
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        
        #Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        
        if args.mode in ["clip"]:
            pos_enabled = False
            neg_enabled = False

        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        end = time.time()
        #Test-time adaptation
        for idx, (images, target) in enumerate(loader):
            infer_ori_image = args.datasets in "caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101/ \
            contrast/jpeg/pixelate/elastic/ \
            defocus/glass/zoom/motion/ \
            saturate/ \
            gaussian/shot/impulse/speckle/ \
            fog/frost/snow/brightness"
            
            # infer_ori_image = True
            image_features, clip_logits, loss, prob_map, pred, ori_feat, ori_output = get_clip_logits(images, clip_model, clip_weights, infer_ori_image=infer_ori_image)
            target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)

            
            if pos_enabled:
                if args.mode in ["tda"]:
                    update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'], fifo=False)
                else:
                    update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'], fifo=False)
                    
                
                if args.mode in ["boostadapter"]:
                    select_feat, select_output, select_idx = select_confident_samples(ori_feat, ori_output, 0.1)
                    select_entropy = get_output_entropy(select_output)
                    
                    cur_pos_cache = copy.deepcopy(pos_cache)
                    for i in range(select_entropy.shape[0]):
                        cur_pred = int(select_output[i].argmax(dim=-1).item())
                        cur_feat = select_feat[i]
                        update_cache(cur_pos_cache, cur_pred, [cur_feat.unsqueeze(0), select_entropy[i].item()], pos_params['shot_capacity'] + cfg['delta'], fifo=False)

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)

            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                if args.mode in ["tda"]:
                    final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
                elif args.mode in ["boostadapter"]:
                    final_logits += compute_cache_logits(image_features, cur_pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)

            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))

                
            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)
            end = time.time()
            
            top1.update(acc, 1)
            batch_time.update(time.time() - end, 1)
            end = time.time()

            if idx%1==0:
                progress.display(idx, logger)
        progress.display_summary(logger)
        
        return sum(accuracies)/len(accuracies)



def main():
    torch.set_num_threads(8)
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    model_path = args.backbone
    clip_model, preprocess = clip.load(model_path)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)
    
    fmt = "[{time: MM-DD hh:mm:ss}] {message}"
    config = {
        "handlers": [
            {"sink": sys.stderr, "format": fmt},
        ],
    }
    logger.configure(**config)
    
    if "search" in args.exp_name:
        file_root = f"log/search/{args.datasets}/"
    else:
        file_root = f"log/eval/{args.datasets}/"
    file_path = f"{file_root}/{args.exp_name}.txt"
    if not os.path.exists(file_root):
        os.makedirs(file_root)
    if os.path.exists(file_path):
        print("Experiment exists. Skipping...")
        exit()
    open(file_path,'w').close()
    logger.add(file_path,format=fmt)
    
    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        logger.info(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        logger.info(args)
        logger.info("\nRunning dataset configurations:")
        logger.info(cfg)
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess, args)
        clip_weights = clip_classifier(classnames, template, clip_model)

        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, logger, args, cfg)

if __name__ == "__main__":
    main()