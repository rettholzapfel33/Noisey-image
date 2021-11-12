import torch
import sys
import os
sys.path.append('../')
from src.predict_img import process_img, predict_img, load_model_from_cfg, visualize_result, transparent_overlays, get_color_palette

_model = load_model_from_cfg(os.path.join('../src/config/ade20k-hrnetv2.yaml'))
_input = torch.Tensor(1,3,416,416)
batch_input = {'img_data': _input}
traced_model = torch.jit.trace(_model, batch_input)
traced_model.save('mitseg_ade20k_hrnetv2.pt')