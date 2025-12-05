import argparse
import os
import mmcv
from mmcv import Config, DictAction, imdenormalize
from mmdet.datasets import build_dataloader
import torch.distributed as dist
from mmrotate.datasets import build_dataset
from mmdet.apis import init_random_seed, set_random_seed
import numpy as np
from mmcv.visualization import  color_val
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default='/T2007061/gaoke/LSKNet/configs/lsknet/lsk_s_fpn_1x_dota_le90.py', help='train config file path')
    parser.add_argument('--work-dir',default='/T2007061/gaoke/LSKNet/work_dirs/lsk_s_fpn_1x_dota_le90', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    '''
    {'type': 'DOTADataset', 'ann_file': 'data/split_1024_dota1_0/trainval/annfiles/', 'img_prefix': 'data/split_1024_dota1_0/trainval/images/', 'pipeline': [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'type': 'RResize', 'img_scale': (1024, 1024)}, {'type': 'RRandomFlip', 'flip_ratio': [0.25, 0.25, 0.25], 'direction': ['horizontal', 'vertical', 'diagonal'], 'version': 'le90'}, {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, {'type': 'Pad', 'size_divisor': 32}, {'type': 'DefaultFormatBundle'}, {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}], 'version': 'le90'}
    '''
    datasets = build_dataset(cfg.data.train)
    print(datasets)
    print('datasets build finish!')

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    # set gpu_ids
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    data_loaders = build_dataloader(dataset=datasets,
                                    samples_per_gpu=2,
                                    workers_per_gpu=2,
                                    num_gpus=len(cfg.gpu_ids),
                                    dist=False,
                                    seed=cfg.seed,
                                    runner_type=runner_type,
                                    persistent_workers=False
                                    )
    print('data_loaders build finish!')
    
    for i, data_batch in enumerate(data_loaders):   # data_batch ['img_metas', 'img', 'gt_bboxes', 'gt_labels']
        img_metas_batch = data_batch['img_metas'].data[0]   # len = 2
        img_batch = data_batch['img'].data[0]               # [2, 3, 1024, 1024]
        gt_bboxes_batch = data_batch['gt_bboxes'].data[0]       # ([n, 5], [n, 5]) = ([n, [x, y, w, h, angle]], [n, [x, y, w, h, angle]])
        gt_labels_batch = data_batch['gt_labels'].data[0]       # ([n], [n])
        mean = np.array(cfg.img_norm_cfg['mean'])
        std = np.array(cfg.img_norm_cfg['std'])

        for j in range(len(img_metas_batch)):
            ori_filename = str(img_metas_batch[j]['ori_filename'])
            print(ori_filename)
            bboxes = np.array(gt_bboxes_batch[j])
            labels = np.array(gt_labels_batch[j])
            img_hwc = np.transpose(img_batch[j].numpy(), [1, 2, 0])
            img_numpy_float = imdenormalize(img_hwc, mean, std)
            img_numpy_uint8 = np.array(img_numpy_float, np.uint8)

            assert bboxes.ndim == 2
            assert labels.ndim == 1
            assert bboxes.shape[0] == labels.shape[0]
            assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
            img = np.ascontiguousarray(img_numpy_uint8)

            score_thr = 0
            bbox_color = 'green'
            text_color = 'green'
            class_names = None
            font_scale = 0.5

            if score_thr > 0:
                assert bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > score_thr
                bboxes = bboxes[inds, :]
                labels = labels[inds]

            bbox_color = color_val(bbox_color)
            text_color = color_val(text_color)

            for bbox, label in zip(bboxes, labels):
                bbox_int = bbox.astype(np.int32)
                obb = [[bbox[0], bbox[1]], [bbox[2], bbox[3]], bbox[4]*57.2957]
                obb = np.int0(cv2.boxPoints(obb))
                cv2.drawContours(img, [obb], 0, bbox_color)

                label_text = class_names[
                    label] if class_names is not None else f'cls {label}'
                if len(bbox) > 4:
                    label_text += f'|{bbox[-1]:.02f}'
                cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                            cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
                mmcv.imwrite(img, ori_filename)

        if i == 0:
            break


if __name__ == '__main__':
    main()
