#!/usr/bin/env python3
"""Safely add geometry/RGB-color ablation switches to LLVIP stage-1 training files.
Run once from /root/autodl-tmp/LLVIP_Project with: python prepare_ablation.py
"""
from pathlib import Path
from datetime import datetime
import ast

ROOT = Path.cwd()
augment = ROOT / 'code/fusion_augment.py'
dataset = ROOT / 'code/fusion_dataset.py'
train = ROOT / 'code/train_fusion_stage1.py'
paths = [augment, dataset, train]

for path in paths:
    if not path.is_file():
        raise SystemExit(f'找不到文件: {path}。请在 LLVIP_Project 根目录执行。')

old = {path: path.read_text(encoding='utf-8') for path in paths}

# Do not partially modify anything on a repeated run.
if ('--no-geo-aug' in old[train] and '--no-color-aug' in old[train]
    and 'geo_aug=True, color_aug=True' in old[dataset]
    and 'geo_aug=True, color_aug=True' in old[augment]):
    print('检测到消融开关已全部配置，无需重复修改。')
    raise SystemExit(0)


def replace_one(text, before, after, desc):
    found = text.count(before)
    if found != 1:
        raise RuntimeError(f'{desc} 位置应出现1次，实际{found}次；未修改任何文件。')
    return text.replace(before, after, 1)

s1 = old[augment]
s1 = replace_one(
    s1, 'def augment_pair(rgb, ir, labels, size=640):',
    'def augment_pair(rgb, ir, labels, size=640, geo_aug=True, color_aug=True):',
    '增强函数参数')
# Two geometric transformations: HFlip and affine
old_geo = 'if np.random.random() < 0.5:'
if s1.count(old_geo) != 2:
    raise RuntimeError('期望恰好找到2处几何增强条件；未修改任何文件。')
s1 = s1.replace(old_geo, 'if geo_aug and np.random.random() < 0.5:')
s1 = replace_one(s1, 'if np.random.random() < 0.8:',
                 'if color_aug and np.random.random() < 0.8:', 'RGB HSV颜色增强')
s1 = replace_one(s1, 'if np.random.random() < 0.3:',
                 'if color_aug and np.random.random() < 0.3:', 'RGB亮度对比度增强')

s2 = old[dataset]
s2 = replace_one(
    s2, 'def __init__(self, root, split="train", imgsz=640):',
    'def __init__(self, root, split="train", imgsz=640, geo_aug=True, color_aug=True):',
    '数据集构造参数')
s2 = replace_one(
    s2, '        self.augment = (split == "train")',
    '        self.augment = (split == "train")\n'
    '        self.geo_aug = bool(geo_aug) and self.augment\n'
    '        self.color_aug = bool(color_aug) and self.augment',
    '训练集增强开关')
s2 = replace_one(
    s2, '                rgb, ir, labels, self.imgsz\n',
    '                rgb, ir, labels, self.imgsz,\n'
    '                geo_aug=self.geo_aug, color_aug=self.color_aug\n',
    '调用增强函数')

s3 = old[train]
s3 = replace_one(
    s3, '    return parser.parse_args()',
    '    parser.add_argument("--no-geo-aug", action="store_true",\n'
    '                        help="消融：关闭 RGB-IR 同步几何增强")\n'
    '    parser.add_argument("--no-color-aug", action="store_true",\n'
    '                        help="消融：关闭 RGB 颜色/亮度/对比度增强")\n'
    '    return parser.parse_args()',
    '训练命令行参数')
s3 = replace_one(
    s3, 'root=args.data_root, split="train", imgsz=args.imgsz',
    'root=args.data_root, split="train", imgsz=args.imgsz,\n'
    '        geo_aug=not args.no_geo_aug, color_aug=not args.no_color_aug',
    '训练数据集构造')

# Validate all three modifications before writing anything.
for path, content in [(augment, s1), (dataset, s2), (train, s3)]:
    ast.parse(content, filename=str(path))

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
for path in paths:
    backup = path.with_name(path.name + '.pre_ablation_' + stamp + '.bak')
    backup.write_text(old[path], encoding='utf-8')
    print(f'已备份：{backup.relative_to(ROOT)}')

for path, content in [(augment, s1), (dataset, s2), (train, s3)]:
    path.write_text(content, encoding='utf-8')
    print(f'修改成功：{path.relative_to(ROOT)}')

print('完成：默认保留原来的完整增强；--no-geo-aug 和 --no-color-aug 可以独立关闭。')
