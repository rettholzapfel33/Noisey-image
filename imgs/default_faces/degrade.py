import cv2
import os
import sys
sys.path.append('../../../')
from src import transforms
import numpy as np
import random
random.seed(10)
np.random.seed(10)

def runDegrade(path,func, params, out):
    img = cv2.imread(path)
    frames = [img]
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"MP4V"), 1, (img.shape[1], img.shape[0]))
    writer.write(img)

    for p in params:
        pre_img = np.copy(img)
        noise_img = func(pre_img, p)
        cv2.imshow('test', noise_img)
        cv2.waitKey(100)
        frames.append(noise_img)
        writer.write(noise_img)
    writer.release()

if __name__ == '__main__':
    runDegrade('100faces.jpg', transforms.saltAndPapper_noise, [0.1, 0.2, 0.3, 0.4, 0.5], '100faces_degrade_sp.mp4')
    runDegrade('worlds-largest-selfie.jpg', transforms.saltAndPapper_noise, [0.1, 0.2, 0.3, 0.4, 0.5], 'worlds-largest-selfie_degrade_sp.mp4')

    runDegrade('100faces.jpg', transforms.jpeg_comp, [75, 50, 25, 10, 5, 3, 1], '100faces_degrade_jpeg.mp4')
    runDegrade('worlds-largest-selfie.jpg', transforms.jpeg_comp, [75, 50, 25, 10, 5, 3, 1], 'worlds-largest-selfie_degrade_jpeg.mp4')
    