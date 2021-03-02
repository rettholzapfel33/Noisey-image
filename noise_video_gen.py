import os
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt


# updated sp noise
def saltAndPapper_noise(image, prob=0.01):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

'''
# old sp noise (red channel)
def saltAndPapper_noise(image, s_vs_p=0.5, amount=0.05):
    row,col,ch = image.shape
    # s_vs_p = 0.5
    # amount = 0.05
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out
'''

def gauss_noise(image, mean=0, var=0.1):
    row,col,ch= image.shape
    # mean = 0
    # var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy

def poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy

def speckle_noise(image):
    row,col,ch = image.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)
    noisy = image + image * gauss
    return noisy


def start(img_path):
    img = cv2.imread(img_path)
    frame_width = img.shape[1]
    frame_height = img.shape[0]
    out = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (frame_width, frame_height))
    org_img = img.copy()
    amount = 0.01
    while(True):
        #ret, frame = cap.read()
        frame = saltAndPapper_noise(org_img, amount)
        amount+=0.01

        frame = cv2.putText(frame, "salt & papper", (int(frame_width*0.80),50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "amount = {:.5f}".format(amount), (int(frame_width*0.80),100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

        #if ret == True:

        # Write the frame into the file 'output.avi'
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame',frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Break the loop
        #else:
        #    break

    # When everything done, release the video capture and video write objects
    # cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Apply noise to an image and write to an mp4 file")
    parser.add_argument("-i", "--img", required=True, type=str, metavar='', help="an image path")
    parser.add_argument("-s", "--save", default="out.mp4", type=str, metavar='', help="save video path")
    args = parser.parse_args()

    # Create a VideoCapture object
    # cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    #if (cap.isOpened() == False):
    #    print("Unable to read camera feed")

    # img = cv2.imread('car_detection_sample1.png')
    img = cv2.imread(args.img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    #frame_width = int(cap.get(3)+10)
    #frame_height = int(cap.get(4))
    frame_width = img.shape[1]
    frame_height = img.shape[0]

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width, frame_height))
    out = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (frame_width, frame_height))

    org_img = img.copy()
    amount = 0.01
    while(True):
        #ret, frame = cap.read()
        # frame = saltAndPapper_noise(org_img, s_vs_p=0.5, amount=amount)
        frame = saltAndPapper_noise(org_img, amount)
        amount+=0.01

        frame = cv2.putText(frame, "salt & papper", (int(frame_width*0.80),50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "amount = {:.5f}".format(amount), (int(frame_width*0.80),100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

        #if ret == True:

        # Write the frame into the file 'output.avi'
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame',frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Break the loop
        #else:
        #    break

    # When everything done, release the video capture and video write objects
    # cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
