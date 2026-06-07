_base_ = ['./fcn_hr48-M2MRF-C_40k_idrid_baseline_copy.py']

# Scratch training recipe that preserves the official IDRiD baseline while
# enabling only weighted binary dice loss under the verified single-GPU runtime
# setup: bs=3 + fp16.

data = dict(samples_per_gpu=3, workers_per_gpu=0)

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

work_dir = 'work_dirs/la_m2mrf/idrid_weighted_dice_scratch_bs3_fp16'
