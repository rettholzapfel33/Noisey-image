import os
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from noises import *

def add_noise_img(img, level, save):
    img = cv2.imread(img)
   # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frame_width = img.shape[1]
    frame_height = img.shape[0]

    org_img = img.copy()
    amount = level
    
    frame = saltAndPapper_noise(org_img, amount)
    
    frame = cv2.putText(frame, "salt & papper", (int(frame_width*0.80),50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "amount = {:.5f}".format(amount), (int(frame_width*0.80),100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        
    #cv2.imshow('frame', frame)
    cv2.imwrite(save, frame)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Apply noise to an image and write to an image file")
    parser.add_argument("-i", "--img", required=True, type=str, metavar='', help="an image path")
    parser.add_argument("-l", "--level", default=0.001, type=float, metavar='', help="noise level")
    parser.add_argument("-s", "--save", default="out_img.jpg", type=str, metavar='', help="save image path")
    
    args = parser.parse_args()

    
    img = cv2.imread(args.img)
   # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    frame_width = img.shape[1]
    frame_height = img.shape[0]

    org_img = img.copy()
    amount = args.level
    
    frame = saltAndPapper_noise(org_img, amount)
    
    frame = cv2.putText(frame, "salt & papper", (int(frame_width*0.80),50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "amount = {:.5f}".format(amount), (int(frame_width*0.80),100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        
    cv2.imshow('frame', frame)
    cv2.imwrite(args.save, frame)
    
    '''
    while(True):
        #ret, frame = cap.read()
        # frame = saltAndPapper_noise(org_img, s_vs_p=0.5, amount=amount)
        frame = saltAndPapper_noise(org_img, amount)
        amount+=0.001

        

        #if ret == True:

        # Write the frame into the file 'output.avi'
        out.write(frame)

        # Display the resulting frame
        

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Break the loop
        #else:
        #    break

    # When everything done, release the video capture and video write objects
    # cap.release()
    '''

    # Closes all the frames
    # cv2.destroyAllWindows()
