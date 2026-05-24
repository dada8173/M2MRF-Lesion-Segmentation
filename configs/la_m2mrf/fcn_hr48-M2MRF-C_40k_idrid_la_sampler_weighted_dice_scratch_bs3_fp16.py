_base_ = ['./fcn_hr48-M2MRF-C_40k_idrid_baseline_copy.py']

# Scratch training recipe that follows the official IDRiD baseline schedule
# while adding the LA sampler + weighted dice modifications under the verified
# single-GPU runtime setup: bs=3 + fp16.

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

data = dict(
    samples_per_gpu=3, workers_per_gpu=0, train=dict(pipeline=train_pipeline))

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='WeightedBinaryDiceLoss',
            loss_type='dice',
            class_weight=[1.0, 1.0, 1.5, 2.0],
            smooth=1e-5,
            loss_weight=1.0)))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=5000, metric='mIoU')

work_dir = 'work_dirs/la_m2mrf/idrid_la_sampler_weighted_dice_scratch_bs3_fp16'
