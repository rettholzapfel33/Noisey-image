from src.transforms import mainAug
import argparse
import cv2
from src import models

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Default image maker for Noisey Image GUI.")
    parser.add_argument("-i", "--img", required=True, type=str, metavar='', help="an image path")
    
    args = parser.parse_args()

    original = cv2.imread(args.img)

    augmented = original.copy()

    mainAug.append('Gaussian Noise')
    mainAug.append('JPEG Compression')
    #mainAug.append('Salt and Pepper')
    for aug in mainAug:
        augmented = aug(augmented, example=True)

    #model = models._registry["Semantic Segmentation"]
    model = models._registry["Face Detection (YOLOv3)"]
    print("Initialized")
    model.conf_thres = 0.2 # added for yolov4
    model.initialize()
    pred = model.run(augmented)
    result = model.draw(pred, augmented)
    model.deinitialize()

    cv2.imwrite("imgs/default_imgs/original.png", original)
    #cv2.imwrite("imgs/default_imgs/segmentation.png", result["segmentation"])
    cv2.imwrite("imgs/default_imgs/segmentation_overlay.png", result["dst"])
    #cv2.imwrite("imgs/default_imgs/segmentation_overlay.png", result)