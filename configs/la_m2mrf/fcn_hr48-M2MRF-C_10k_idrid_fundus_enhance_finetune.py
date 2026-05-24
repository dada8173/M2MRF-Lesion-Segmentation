_base_ = ['./fcn_hr48-M2MRF-C_10k_idrid_finetune_control.py']

img_norm_cfg = dict(
    mean=[116.513, 56.437, 16.309], std=[80.206, 41.232, 13.293], to_rgb=True)
image_scale = (1440, 960)
crop_size = (960, 1440)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='FundusCLAHEEnhancement',
         apply_prob=1.0,
         mode='green',
         clip_limit=2.0,
         tile_grid_size=(8, 8)),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=image_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='FundusCLAHEEnhancement',
                 apply_prob=1.0,
                 mode='green',
                 clip_limit=2.0,
                 tile_grid_size=(8, 8)),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
work_dir = 'work_dirs/la_m2mrf/idrid_fundus_enhance_finetune'
