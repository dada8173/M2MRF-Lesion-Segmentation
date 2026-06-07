_base_ = ['./fcn_hr48-M2MRF-C_10k_idrid_finetune_control.py']

img_norm_cfg = dict(
    mean=[116.513, 56.437, 16.309], std=[80.206, 41.232, 13.293], to_rgb=True)
image_scale = (1440, 960)
crop_size = (960, 1440)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=image_scale, ratio_range=(0.5, 2.0)),
    dict(
        type='LesionAwareRandomCrop',
        crop_size=crop_size,
        lesion_prob=0.7,
        small_lesion_prob=0.4,
        target_classes=[1, 2, 3, 4],
        priority_classes=[3, 4],
        cat_max_ratio=0.75,
        num_retry=10),
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(train=dict(pipeline=train_pipeline))

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='WeightedBinaryDiceLoss',
            loss_type='dice',
            class_weight=[1.0, 1.0, 1.5, 2.0],
            smooth=1e-5,
            loss_weight=1.0)))

work_dir = 'work_dirs/la_m2mrf/idrid_la_sampler_weighted_dice_finetune'
