_base_ = ['./fcn_hr48-M2MRF-C_40k_idrid_baseline_copy.py']

# Official-checkpoint continuation control for LA-M2MRF ablations.
# The current mainline schedule is shortened to 5k for faster ablation cycles.
load_from = '/mnt/c/Users/dachen/Downloads/fcn_hr48-M2MRF-C_40k_idrid_bdice_iter_40000.pth'

data = dict(samples_per_gpu=3, workers_per_gpu=0)

optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=5000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500, metric='mIoU')

work_dir = 'work_dirs/la_m2mrf/idrid_finetune_control'
