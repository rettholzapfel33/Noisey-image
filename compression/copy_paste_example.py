import os
import cv2
import numpy as np

if __name__ == '__main__':
    
    # EXAMPLE OF COPYING AND PASTE PORTIONS OF ONE IMAGE TO ANOTHER:
    # load in image:
    image1 = cv2.imread('dog.jpg')
    
    # make a dummy destination black image:
    image2 = np.ndarray(image1.shape, dtype=np.uint8) # image1.shape <---- dimensions of the image (height, width, channel)
    # uint8 is unsigned integer 8 (0-255). Needed for a traditional image!

    # make some random box coordinate from center:
    dummy_box = [image1.shape[0]//2, image1.shape[1]//2, (image1.shape[0]//2)+100, (image1.shape[1]//2)+100] # x1, y1, x2, y2

    '''
    Dummy box would look like this:
    --------------------------------
    |                              |
    |                              |
    |      (x1,y1) *_____          |
    |              |    |          |
    |              |____|          |
    |                    * (x2,y2) |
    |______________________________|
    '''

    # cut out dummybox from image1 and put in image2:
    image_chunk = image1[dummy_box[0]:dummy_box[2], dummy_box[1]:dummy_box[3],:] # x1:x2, y1:y2, all three color channels
    image2[dummy_box[0]:dummy_box[2], dummy_box[1]:dummy_box[3],:] = image_chunk # plunk this little guy into the new black canvas

    # display the following in opencv2:
    cv2.imshow('image1', image1)
    cv2.imshow('imagechunk', image_chunk)
    cv2.imshow('image2', image2)
    cv2.waitKey(-1)
    # click any key to close all windows!