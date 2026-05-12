from collections import defaultdict
from glob import glob
from itertools import chain
from tqdm import tqdm
import argparse
import json
import os
import torch

from scipy import stats
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from industrial.cpr.dataset import DATASET_INFOS, read_image, read_mask, test_transform
from industrial.cpr.metrics import compute_ap_torch, compute_pixel_auc_torch, compute_pro_torch, compute_image_auc_torch
from industrial.cpr.models import create_model, MODEL_INFOS, CPR
from industrial.cpr.utils import fix_seeds


def get_args_parser():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("-dn", "--dataset-name", type=str, default="mvtec", choices=["mvtec", "mvtec_3d", "btad"], help="dataset name")
    parser.add_argument("-ss", "--scales", type=int, nargs="+", help="multiscale", default=[4, 8])
    parser.add_argument("-kn", "--k-nearest", type=int, default=10, help="k nearest")
    parser.add_argument("-r", "--resize", type=int, default=320, help="image resize")
    parser.add_argument("-fd", "--foreground-dir", type=str, default=None, help="foreground dir")
    parser.add_argument("-rd", "--retrieval-dir", type=str, default='log/retrieval_mvtec_DenseNet_features.denseblock1_320', help="retrieval dir")
    parser.add_argument("--sub-categories", type=str, nargs="+", default=None, help="sub categories", choices=list(chain(*[x[0] for x in list(DATASET_INFOS.values())])))
    parser.add_argument("--T", type=int, default=512)  # for image-level inference
    parser.add_argument("-rs", "--region-sizes", type=int, nargs="+", help="local retrieval region size", default=[3, 1])
    parser.add_argument("-pm", "--pretrained-model", type=str, default='DenseNet', choices=list(MODEL_INFOS.keys()), help="pretrained model")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None, help="checkpoints")
    parser.add_argument("--save-maps", action="store_true", help="Save anomaly heatmaps during test")
    parser.add_argument("--save-scores", action="store_true", help="Save per-image anomaly scores as CSV")
    parser.add_argument("--save-dir", type=str, default="./saved_results", help="Directory to save results")
    parser.add_argument("--bh-fdr", type=float, default=None, help="FDR rate for Benjamini-Hochberg thresholding (e.g. 0.05)")
    parser.add_argument("--validation", action="store_true", help="Run on validation/good images and save heatmaps for EVT fitting")
    parser.add_argument("--data-root", type=str, default=None, help="dataset root dir (default: ./data/{dataset_name})")
    parser.add_argument("--val-dir", type=str, default=None, help="Override validation image directory (e.g., augmented val set)")
    return parser

@torch.no_grad()
def test(model: CPR, train_fns, test_fns, retrieval_result, foreground_result, resize, region_sizes, root_dir, knn, T, save_maps=False, save_scores=False, save_dir=None, category=None, bh_fdr=None, gt_base=None):
    model.eval()
    train_local_features = [torch.zeros((len(train_fns), out_channels, *shape[2:]), device='cuda') for shape, out_channels in zip(model.backbone.shapes, model.lrb.out_channels_list)]
    train_foreground_weights = []
    k2id = {}
    for idx, image_fn in enumerate(tqdm(train_fns, desc='extract train local features', leave=False)):
        k = os.path.relpath(image_fn, root_dir)
        image = read_image(image_fn, (resize, resize))
        image_t = test_transform(image)
        features_list, ori_features_list = model(image_t[None].cuda())
        for i, features in enumerate(features_list):
            train_local_features[i][idx:idx+1] = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8)
        if k in foreground_result:
            train_foreground_weights.append(torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda())
        k2id[k] = idx
    if train_foreground_weights:
        train_foreground_weights = torch.stack(train_foreground_weights)

    # Build null distribution from training images for BH thresholding
    null_mean, null_std = None, None
    if bh_fdr is not None:
        null_scores = []
        for idx, image_fn in enumerate(tqdm(train_fns, desc='build null distribution', leave=False)):
            k = os.path.relpath(image_fn, root_dir)
            if k not in retrieval_result:
                continue
            retrieval_idxs = [k2id[rk] for rk in retrieval_result[k][:knn] if rk in k2id]
            if not retrieval_idxs:
                continue
            features_list_i = [train_local_features[s][idx:idx+1] for s in range(len(region_sizes))]
            retrieval_features_list_i = [train_local_features[s][retrieval_idxs] for s in range(len(region_sizes))]
            train_scores = []
            for features, retrieval_features, region_size in zip(features_list_i, retrieval_features_list_i, region_sizes):
                unfold = nn.Unfold(kernel_size=region_size, padding=region_size // 2)
                region_features = unfold(retrieval_features).reshape(retrieval_features.shape[0], retrieval_features.shape[1], -1, retrieval_features.shape[2], retrieval_features.shape[3])
                dist = (1 - (features[:, :, None] * region_features).sum(1))
                dist = dist / (unfold(torch.ones(1, 1, retrieval_features.shape[2], retrieval_features.shape[3], device=retrieval_features.device)).reshape(1, -1, retrieval_features.shape[2], retrieval_features.shape[3]) + 1e-8)
                score = dist.min(1)[0].min(0)[0]
                score = F.interpolate(score[None, None], size=(features_list_i[0].shape[2], features_list_i[0].shape[3]), mode="bilinear", align_corners=False)[0, 0]
                train_scores.append(score)
            score = torch.stack(train_scores).sum(0)
            score_g = gaussian_blur(score[None], (33, 33), 4)[0]
            null_scores.append(score_g.flatten().cpu().numpy())
        null_scores = np.concatenate(null_scores)
        null_mean = null_scores.mean()
        null_std = null_scores.std()
        print(f'  Null distribution: mean={null_mean:.6f}, std={null_std:.6f}, n={len(null_scores)}')
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f'{category}_null_scores.npy'), null_scores)

    gts = []
    i_gts = []
    preds = defaultdict(list)
    for image_fn in tqdm(test_fns, desc='predict test data', leave=False):
        image = read_image(image_fn, (resize, resize))
        image_t = test_transform(image)
        k = os.path.relpath(image_fn, root_dir)
        image_name = os.path.basename(k)[:-4]
        anomaly_name = os.path.dirname(k).rsplit('/', 1)[-1]
        _gt_base = gt_base if gt_base is not None else os.path.join(root_dir, 'ground_truth')
        mask_fn = os.path.join(_gt_base, anomaly_name, image_name + '_mask.png')
        if os.path.exists(mask_fn):
            mask = read_mask(mask_fn, (resize, resize))
        else:
            mask = np.zeros((resize, resize))
        
        gts.append((mask > 127).astype(int))
        i_gts.append((mask > 127).sum() > 0 and 1 or 0)
        
        features_list, ori_features_list = model(image_t[None].cuda())
        features_list = [features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8) for features in features_list]
        retrieval_idxs = [k2id[retrieval_k] for retrieval_k in retrieval_result[k][:knn]]
        retrieval_features_list = [train_local_features[i][retrieval_idxs] for i in range(len(features_list))]
        
        scores = []
        assert len(features_list) == len(retrieval_features_list) == len(region_sizes)
        for features, retrieval_features, region_size in zip(features_list, retrieval_features_list, region_sizes):
            unfold = nn.Unfold(kernel_size=region_size, padding=region_size // 2)
            region_features = unfold(retrieval_features).reshape(retrieval_features.shape[0], retrieval_features.shape[1], -1, retrieval_features.shape[2], retrieval_features.shape[3])  # b x c x r^2 x h x w
            dist = (1 - (features[:, :, None] * region_features).sum(1))  # b x r^2 x h x w
            # fill position is set to a large value
            dist = dist / (unfold(torch.ones(1, 1, retrieval_features.shape[2], retrieval_features.shape[3], device=retrieval_features.device)).reshape(1, -1, retrieval_features.shape[2], retrieval_features.shape[3]) + 1e-8)
            score = dist.min(1)[0].min(0)[0]
            score = F.interpolate(
                score[None, None],
                size=(features_list[0].shape[2], features_list[0].shape[3]),
                mode="bilinear", align_corners=False
            )[0, 0]
            scores.append(score)
        score = torch.stack(scores).sum(0)
        score = F.interpolate(
            score[None, None],
            size=(mask.shape[0], mask.shape[1]),
            mode="bilinear", align_corners=False
        )[0, 0]
        if k in foreground_result:
            foreground_weight = torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda()
            foreground_weight = torch.cat([foreground_weight[None], train_foreground_weights[retrieval_idxs]]).max(0)[0]
            score = score * foreground_weight
        score_g = gaussian_blur(score[None], (33, 33), 4)[0]  # PatchCore
        det_score = torch.topk(score_g.flatten(), k=T)[0].sum()  # DeSTSeg
        preds['i'].append(det_score)
        preds['p'].append(score_g)
    gts = torch.from_numpy(np.stack(gts)).cuda()
    metrics = {
        'pro': compute_pro_torch(gts, torch.stack(preds['p'])),
        'ap': compute_ap_torch(gts, torch.stack(preds['p'])),
        'pixel-auc': compute_pixel_auc_torch(gts, torch.stack(preds['p'])),
        'image-auc': compute_image_auc_torch(torch.tensor(i_gts).long().cuda(), torch.stack(preds['i'])),
    }

    if save_dir and (save_maps or save_scores):
        import csv
        from matplotlib import pyplot as plt
        from sklearn.metrics import precision_recall_curve

        all_scores = [s.item() for s in preds['i']]
        all_labels = i_gts

        # Save heatmaps
        if save_maps:
            for idx, image_fn in enumerate(test_fns):
                k = os.path.relpath(image_fn, root_dir)
                image_name = os.path.basename(k)[:-4]
                anomaly_name = os.path.dirname(k).rsplit('/', 1)[-1]
                out_dir = os.path.join(save_dir, 'heatmaps', category, anomaly_name)
                os.makedirs(out_dir, exist_ok=True)

                amap = preds['p'][idx].cpu().numpy()
                amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

                input_img = cv.resize(cv.imread(image_fn), (resize, resize))
                input_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
                plt.imsave(os.path.join(out_dir, f'{image_name}_input.png'), input_img)
                plt.imsave(os.path.join(out_dir, f'{image_name}_heatmap.png'), amap_norm, cmap='jet')
                np.save(os.path.join(out_dir, f'{image_name}_heatmap_raw.npy'), amap)

                amap_color = (plt.cm.jet(amap_norm)[:, :, :3] * 255).astype(np.uint8)
                overlay = cv.addWeighted(input_img, 0.5, amap_color, 0.5, 0)
                plt.imsave(os.path.join(out_dir, f'{image_name}_overlay.png'), overlay)

                if i_gts[idx] == 1:
                    gt_map = gts[idx].cpu().numpy()
                    plt.imsave(os.path.join(out_dir, f'{image_name}_gt.png'), gt_map, cmap='gray')
                plt.close('all')
            print(f'  Heatmaps saved to {save_dir}/heatmaps/{category}/')

        # Save scores CSV
        if save_scores:
            # Find best threshold
            precs, recs, thrs = precision_recall_curve(all_labels, all_scores)
            f1s = 2 * precs * recs / (precs + recs + 1e-7)
            best_thr = thrs[np.argmax(f1s[:-1])]

            scores_dir = os.path.join(save_dir, 'scores')
            os.makedirs(scores_dir, exist_ok=True)
            csv_path = os.path.join(scores_dir, f'{category}_scores.csv')
            with open(csv_path, 'w', newline='') as f:
                f.write(f"# Metrics: {', '.join(f'{k}={v:.4f}' for k, v in metrics.items())}\n")
                writer = csv.DictWriter(f, fieldnames=['filename', 'defect_type', 'anomaly_score', 'ground_truth', 'predicted'])
                writer.writeheader()
                for idx, image_fn in enumerate(test_fns):
                    k = os.path.relpath(image_fn, root_dir)
                    image_name = os.path.basename(k)
                    anomaly_name = os.path.dirname(k).rsplit('/', 1)[-1]
                    score = all_scores[idx]
                    writer.writerow({
                        'filename': image_name,
                        'defect_type': anomaly_name,
                        'anomaly_score': score,
                        'ground_truth': 'anomaly' if all_labels[idx] == 1 else 'normal',
                        'predicted': 'anomaly' if score >= best_thr else 'normal',
                    })
            print(f'  Scores saved to {csv_path}')

    return metrics

@torch.no_grad()
def validate(model, train_fns, val_fns, retrieval_result, foreground_result, resize, region_sizes, root_dir, knn, save_dir, category):
    """Run model on validation/good images and save heatmaps only (no metrics)."""
    from matplotlib import pyplot as plt
    model.eval()
    train_local_features = [torch.zeros((len(train_fns), out_channels, *shape[2:]), device='cuda') for shape, out_channels in zip(model.backbone.shapes, model.lrb.out_channels_list)]
    train_foreground_weights = []
    k2id = {}
    for idx, image_fn in enumerate(tqdm(train_fns, desc='extract train local features', leave=False)):
        k = os.path.relpath(image_fn, root_dir)
        image = read_image(image_fn, (resize, resize))
        image_t = test_transform(image)
        features_list, ori_features_list = model(image_t[None].cuda())
        for i, features in enumerate(features_list):
            train_local_features[i][idx:idx+1] = features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8)
        if k in foreground_result:
            train_foreground_weights.append(torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda())
        k2id[k] = idx
    if train_foreground_weights:
        train_foreground_weights = torch.stack(train_foreground_weights)

    out_dir = os.path.join(save_dir, 'val_heatmaps', category, 'good')
    os.makedirs(out_dir, exist_ok=True)
    n_saved = 0

    for image_fn in tqdm(val_fns, desc=f'validation maps: {category}', leave=False):
        image = read_image(image_fn, (resize, resize))
        image_t = test_transform(image)
        k = os.path.relpath(image_fn, root_dir)
        image_name = os.path.basename(k)[:-4]

        features_list, ori_features_list = model(image_t[None].cuda())
        features_list = [features / (torch.norm(features, p=2, dim=1, keepdim=True) + 1e-8) for features in features_list]

        # Use retrieval result if available, otherwise use first K train images
        if k in retrieval_result:
            retrieval_idxs = [k2id[rk] for rk in retrieval_result[k][:knn] if rk in k2id]
        else:
            # Validation images may not have retrieval results — use closest train images
            retrieval_idxs = list(range(min(knn, len(train_fns))))

        retrieval_features_list = [train_local_features[i][retrieval_idxs] for i in range(len(features_list))]

        scores = []
        for features, retrieval_features, region_size in zip(features_list, retrieval_features_list, region_sizes):
            unfold = nn.Unfold(kernel_size=region_size, padding=region_size // 2)
            region_features = unfold(retrieval_features).reshape(retrieval_features.shape[0], retrieval_features.shape[1], -1, retrieval_features.shape[2], retrieval_features.shape[3])
            dist = (1 - (features[:, :, None] * region_features).sum(1))
            dist = dist / (unfold(torch.ones(1, 1, retrieval_features.shape[2], retrieval_features.shape[3], device=retrieval_features.device)).reshape(1, -1, retrieval_features.shape[2], retrieval_features.shape[3]) + 1e-8)
            score = dist.min(1)[0].min(0)[0]
            score = F.interpolate(score[None, None], size=(features_list[0].shape[2], features_list[0].shape[3]), mode="bilinear", align_corners=False)[0, 0]
            scores.append(score)
        score = torch.stack(scores).sum(0)
        score = F.interpolate(score[None, None], size=(resize, resize), mode="bilinear", align_corners=False)[0, 0]
        if k in foreground_result:
            foreground_weight = torch.from_numpy(cv.resize(np.load(foreground_result[k]).astype(float), (resize, resize))).cuda()
            foreground_weight = torch.cat([foreground_weight[None], train_foreground_weights[retrieval_idxs]]).max(0)[0]
            score = score * foreground_weight
        score_g = gaussian_blur(score[None], (33, 33), 4)[0]

        amap = score_g.cpu().numpy()
        amap = cv.resize(amap, (512, 512)).astype(np.float16)
        np.save(os.path.join(out_dir, f'{image_name}_heatmap_raw.npy'), amap)
        n_saved += 1

    print(f'  {category}: {n_saved} validation heatmaps saved to {out_dir}')


def main(args):
    all_categories, object_categories, texture_categories = DATASET_INFOS[args.dataset_name]
    sub_categories = DATASET_INFOS[args.dataset_name][0] if args.sub_categories is None else args.sub_categories
    assert all([sub_category in all_categories for sub_category in sub_categories]), f"{sub_categories} must all be in {all_categories}"
    model_info = MODEL_INFOS[args.pretrained_model]
    layers = [model_info['layers'][model_info['scales'].index(scale)] for scale in args.scales]
    for sub_category_idx, sub_category in enumerate(sub_categories):
        fix_seeds(66)
        model             = create_model(args.pretrained_model, layers).cuda()
        if args.checkpoints is not None:
            checkpoint_fn = args.checkpoints[0] if len(args.checkpoints) == 1 else args.checkpoints[sub_category_idx]
            if '{category}' in checkpoint_fn: checkpoint_fn = checkpoint_fn.format(category=sub_category)
            model.load_state_dict(torch.load(checkpoint_fn), strict=False)
        _data_root = args.data_root or os.path.join('./data', args.dataset_name)
        root_dir = os.path.join(_data_root, sub_category)
        train_fns = sorted(glob(os.path.join(root_dir, 'train/good/*')) or glob(os.path.join(root_dir, 'train/*/*')))
        foreground_result = {}
        with open(os.path.join(args.retrieval_dir, sub_category, 'r_result.json'), 'r') as f:
            retrieval_result = json.load(f)

        if args.validation:
            val_root = getattr(args, 'val_dir', None)
            if val_root:
                val_good = os.path.join(val_root, sub_category, 'validation', 'good')
            else:
                val_good = os.path.join(root_dir, 'validation', 'good')
            val_fns = sorted(glob(os.path.join(val_good, '*')))
            if not val_fns:
                print(f'  {sub_category}: no validation images found, skipping')
                continue
            validate(model, train_fns, val_fns, retrieval_result, foreground_result, args.resize, args.region_sizes, root_dir, args.k_nearest, args.save_dir, sub_category)
        else:
            # Auto-detect AD 2 vs AD 1 test layout
            if os.path.isdir(os.path.join(root_dir, 'test_public')):
                test_dir = os.path.join(root_dir, 'test_public')
                gt_base = os.path.join(test_dir, 'ground_truth')
            else:
                test_dir = os.path.join(root_dir, 'test')
                gt_base = os.path.join(root_dir, 'ground_truth')

            test_fns = sorted(glob(os.path.join(test_dir, '*/*.png')) +
                              glob(os.path.join(test_dir, '*/*.JPG')) +
                              glob(os.path.join(test_dir, '*/*.bmp')))
            # Filter out ground_truth images from test list
            test_fns = [f for f in test_fns if '/ground_truth/' not in f]

            if args.foreground_dir is not None and sub_category in object_categories:
                for fn in train_fns + test_fns:
                    k = os.path.relpath(fn, root_dir)
                    foreground_result[k] = os.path.join(args.foreground_dir, sub_category, os.path.dirname(k), 'f_' + os.path.splitext(os.path.basename(k))[0] + '.npy')
            ret = test(model, train_fns, test_fns, retrieval_result, foreground_result, args.resize, args.region_sizes, root_dir, args.k_nearest, args.T, save_maps=args.save_maps, save_scores=args.save_scores, save_dir=args.save_dir, category=sub_category, bh_fdr=args.bh_fdr, gt_base=gt_base)
            print(f'================={sub_category}=================')
            print(ret)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)