_base_ = './fcn_hr48-M2MRF-C_40k_idrid_full.py'

# Official-checkpoint continuation for full LA-M2MRF with the verified
# bs=3 + fp16 finetune recipe.
load_from = '/mnt/c/Users/dachen/Downloads/fcn_hr48-M2MRF-C_40k_idrid_bdice_iter_40000.pth'

data = dict(samples_per_gpu=3, workers_per_gpu=0)

optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU')

work_dir = 'work_dirs/la_m2mrf/idrid_full_finetune'
