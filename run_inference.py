import os
import cv2
import numpy as np
from mmseg.apis import inference_segmentor, init_segmentor

experiments = {
    "Control": "fcn_hr48-M2MRF-C_10k_idrid_finetune_control",
    "LA-M2MRF-S": "fcn_hr48-M2MRF-C_10k_idrid_la_sampler_finetune",
    "LA-M2MRF-W": "fcn_hr48-M2MRF-C_10k_idrid_weighted_dice_finetune",
    "LA-M2MRF-SW": "fcn_hr48-M2MRF-C_10k_idrid_la_sampler_weighted_dice_finetune"
}

img1_path = "/home/dachen/projects/data/IDRID/image/test/IDRiD_60.jpg"
img2_path = "/home/dachen/projects/data/IDRID/image/test/IDRiD_68.jpg"

def save_mask(model_path, config_path, img_path, out_path):
    model = init_segmentor(config_path, model_path, device='cuda:0')
    result = inference_segmentor(model, img_path)
    # result[0][0] is (4, H, W). We want to convert it to (H, W) where values are 1, 2, 3, 4
    multi_label = np.array(result[0][0])
    # multi_label is 0 or 1.
    final_mask = np.zeros((multi_label.shape[1], multi_label.shape[2]), dtype=np.uint8)
    for c in range(4):
        final_mask[multi_label[c] > 0] = c + 1
    
    cv2.imwrite(out_path, final_mask)

for name, cfg in experiments.items():
    cfg_file = f"configs/la_m2mrf/{cfg}.py"
    ckpt_file = f"work_dirs/la_m2mrf/{cfg}/latest.pth"
    out_dir = f"work_dirs/la_m2mrf/masks/{name}"
    os.makedirs(out_dir, exist_ok=True)
    
    save_mask(ckpt_file, cfg_file, img1_path, os.path.join(out_dir, "IDRiD_60.png"))
    save_mask(ckpt_file, cfg_file, img2_path, os.path.join(out_dir, "IDRiD_68.png"))

print("Generation completed!")
