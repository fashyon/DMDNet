import os
import torch
import MiDaS.utils as utils

import numpy as np

from MiDaS.midas.model_loader import load_model
from MiDaS import run

model_path = './MiDaS/weights/dpt_next_vit_large_384.pt'
model_type = 'dpt_next_vit_large_384'
optimize = False
height = None
square = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)
# input_size = (net_w, net_h)

image_name = './3_1_m.jpg'
# input
original_image_rgb = utils.read_image(image_name)  # in [0, 1]
target_size = original_image_rgb.shape[1::-1]
image = transform({"image": original_image_rgb})["image"]
output_path = './'
side = False
grayscale = False
# compute
with torch.no_grad():
    prediction = run.process(device, model, model_type, image, (net_w, net_h), original_image_rgb.shape[1::-1],
                         optimize, False)
 # output
if output_path is not None:
    filename = os.path.join(
        output_path, os.path.splitext(os.path.basename(image_name))[0] + '-' + model_type
    )
    if not side:
        utils.write_depth(filename, prediction, grayscale, bits=2)
    utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))
