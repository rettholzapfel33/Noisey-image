import sys

import cv2
sys.path.append('./yolov5')
sys.path.append('../')
from src.transforms import AugmentationPipeline, augList
from detect import run
from types import SimpleNamespace
import os

if __name__ == '__main__':
    aug_out_file = "visdrone_val_0"
    names = []

    pipeline = AugmentationPipeline(augList)
    pipeline.append('Gaussian Noise')
    pipeline[0].setParam([10, 15, 20, 25])

    # generate tmp videos:
    for i, arg in enumerate(pipeline[0].args):
        vidName = '%s_%s_%f.mp4'%(aug_out_file, "_".join(pipeline[0].title.split(' ')), arg)
        if not os.path.exists(vidName):
            cap = cv2.VideoCapture('%s.mp4'%(aug_out_file))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            writer = cv2.VideoWriter(
                vidName,
                cv2.VideoWriter_fourcc(*"MP4V"),
                fps,
                (w,h)
            )

            while(True):
                ret, frame = cap.read()
                if not ret: break
                frame = pipeline[0](frame)
                writer.write(frame)
            print("Done %s (%i/%i)"%(vidName, i+1, len(pipeline[0].args)))
        else:
            print("WARNING: %s already exists. Skipping..."%(vidName))
        
        names.append(vidName)

    # non-compounding:
    for name in names: 
        config = SimpleNamespace(
            weights='./weights/best.pt',
            source=name,
            imgsz=(416,416),
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=1000,
            device="",
            view_img=False,
            save_txt=True,
            save_conf=True,
            save_crop=False,
            nosave=False,
            classes=[0],
            agnostic_nms=None,
            augment=False,
            visualize=False,
            update=False,
            project="./",
            name="out_%s"%(name),
            exist_ok=True,
            line_thickness=2,
            hide_labels=True,
            hide_conf=False,
            half=False,
            dnn=False,
        )
        run(**vars(config))