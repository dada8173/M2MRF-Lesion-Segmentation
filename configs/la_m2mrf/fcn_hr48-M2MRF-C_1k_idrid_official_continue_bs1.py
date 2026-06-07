_base_ = ['./fcn_hr48-M2MRF-C_40k_idrid_baseline_copy.py']

# Minimal continuation test from the official IDRiD baseline checkpoint.
# Keep the original baseline method unchanged and only control runtime knobs.
load_from = '/mnt/c/Users/dachen/Downloads/fcn_hr48-M2MRF-C_40k_idrid_bdice_iter_40000.pth'

data = dict(samples_per_gpu=1, workers_per_gpu=0)

optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=1000)
checkpoint_config = dict(by_epoch=False, interval=200)
evaluation = dict(interval=200, metric='mIoU')

work_dir = 'work_dirs/la_m2mrf/idrid_official_continue_bs1'
