import torch
import sys
import os
sys.path.append('../')
from src.obj_detector.detect import load_model
from src.obj_detector.detect import 

_model = load_model(
    #os.path.join('src', 'obj_detector/cfg', 'coco.names'),
    os.path.join('../src', 'obj_detector/cfg', 'yolov3.cfg'),
    os.path.join('../src','obj_detector/weights','yolov3.weights')
)

dummy_input = torch.Tensor(1,3,416,416)
dummy_input = dummy_input.cuda()
traced_module = torch.jit.trace(_model, dummy_input)
print(traced_module)
traced_module.save('yolov3_coco_jit.pt')