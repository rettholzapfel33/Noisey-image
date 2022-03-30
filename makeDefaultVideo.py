from src.transforms import mainAug
import argparse
import cv2
from src import models

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Default video maker for Noisey Image GUI.")
    parser.add_argument("-i", "--video", required=True, type=str, metavar='', help="a video path")
    
    args = parser.parse_args()

    original = cv2.imread(args.img)

    augmented = original.copy()

    mainAug.append('H264 Compression')
    for aug in mainAug:
        augmented = aug(augmented, example=True)

    model = models._registry["Semantic Segmentation"]

    model.initialize()   
    pred = model.run(augmented)
    result = model.draw(pred, augmented)
    model.deinitialize()

    cv2.imwrite("imgs/default_imgs/original.png", original)
    cv2.imwrite("imgs/default_imgs/segmentation.png", result["segmentation"])
    cv2.imwrite("imgs/default_imgs/segmentation_overlay.png", result["dst"])