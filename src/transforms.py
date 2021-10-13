from random import gauss
from src.utils.qt5extra import CheckState
import PyQt5
from PyQt5.QtWidgets import QDialog, QListWidgetItem
from PyQt5 import uic
from cv2 import cv2
import numpy as np
import time
from src.utils import images

def letterbox_image(image, size):
    '''
    Resize image with unchanged aspect ratio using padding.
    This function replaces "letterbox" and enforces non-rectangle static inferencing only
    '''
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h, w, 3), np.uint8) * 114
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image, (nh, nw)


def dim_intensity(image, factor, seed=-1):
    # check if factor is int (constant) or tuple (randomized range with uniform distribution):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if type(factor) == float:
        assert factor <= 1 and factor >= 0
        # adjust value channel:
        value = hsv_img[:, :, 2].astype('float64')*factor
        hsv_img[:, :, 2] = value.astype('uint8')
        image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return image
    elif type(factor) == tuple:
        if seed != -1:
            np.random.seed(seed)
        lower, upper = factor
        assert upper <= 1 and upper >= 0
        assert lower <= 1 and lower >= 0
        assert upper >= lower
        random_factor = np.random.uniform(lower, upper)
        value = hsv_img[:, :, 2].astype('float64')*random_factor
        hsv_img[:, :, 2] = value.astype('uint8')
        image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return image
    else:
        assert False, "factor type needs to be a float or tuple"


def gaussian_noise(image, std, seed=-1):
    mean = 2
    if type(std) == float or type(std) == int:
        assert std > 0
        import matplotlib.pyplot as plt
        # only control standard dev:
        normal_matrix = np.random.normal(mean, std, size=image.shape)
        combined = image+normal_matrix
        np.clip(combined, 0, 255, out=combined)
        return combined.astype('uint8')
    elif type(std) == tuple:
        if seed != -1:
            np.random.seed(seed)
        lower, upper = std
        random_std = np.random.uniform(lower, upper)
        normal_matrix = np.random.normal(mean, random_std, size=image.shape)
        combined = image + normal_matrix
        np.clip(combined, 0, 255, out=combined)
        return combined.astype('uint8')


def gaussian_blur(image, kernel_size_factor, stdX=0, stdY=0, seed=-1):
    if type(kernel_size_factor) == float or type(kernel_size_factor) == int:
        w = (kernel_size_factor*2)+1
        h = (kernel_size_factor*2)+1
        blur_img = cv2.GaussianBlur(
            image, (w, h), cv2.BORDER_DEFAULT, stdX, stdY)
        return blur_img
    if type(kernel_size_factor) == tuple:
        if seed != -1:
            np.random.seed(seed)
        lower, upper = kernel_size_factor
        random_size_factor = np.random.uniform(lower, upper)
        print(random_size_factor)
        w_r = (int(random_size_factor)*2)+1
        h_r = (int(random_size_factor)*2)+1
        print(w_r)
        blur_img = cv2.GaussianBlur(
            image, (w_r, h_r), cv2.BORDER_DEFAULT, stdX, stdY)
        return blur_img


def jpeg_comp(image, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    print(image.shape)
    result, enc_img = cv2.imencode('.jpg', image, encode_param)
    if result is True:
        print(enc_img.shape)
        dec_img = cv2.imdecode(enc_img, 1)
        print(dec_img.shape)
        return dec_img


def normal_comp(image, scale_factor):
    print(image.shape)
    width = int(image.shape[1] * scale_factor / 100)
    height = int(image.shape[0] * scale_factor / 100)
    dim = (width, height)
    # resize image
    resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    print('Resized Dimensions : ', resized_img.shape)
    return resized_img

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

augList = {
    "Intensity": dim_intensity,
    "Gaussian Noise": gaussian_noise,
    "Gaussian Blur": gaussian_blur,
    "JPEG Compression": jpeg_comp,
    "Normal Compression": normal_comp,
    "Salt and Pepper": saltAndPapper_noise,
    "Poisson Noise": poisson_noise,
    "Speckle Noise": speckle_noise,
}

class Augmentation:
    def __init__(self, aug, original_position) -> None:
        self.__run__ = aug
        self.__checked__ = False
        self.__position__ = original_position

    @property
    def enabled(self):
        return self.__checked__

    @property
    def position(self):
        return self.__position__

class AugmentationPipeline:
    def __init__(self, augList:dict) -> None:
        self.__list__ = augList
        self.__augList__ = []
        self.__index__ = 0
        self.__pipeline__ = []

    def __wrapper__(self):
        for pos, item in enumerate(self.__list__.items()):
            _item = Augmentation(item, pos)
            self.__augList__.append(_item)

    def __iter__(self):
        return (x for x in range(len(self.__pipeline__)))

    def __next__(self):
        self.__index__ += 1
        try:
            return self.__pipeline__[self.__index__-1]
        except IndexError:
            self.__index__ = 0
            raise StopIteration

    def __repr__(self) -> str:
        _out = ''
        for pipe in self.__pipeline__:
            _out += '%s\n'%(pipe.position)
        return _out

    def append(self, aug_title):
        augItem = self.__list__.keys().index(aug_title)
        self.__pipeline__.append(augItem)

    def remove(self, aug_title):
        augItem = self.__list__.keys().index(aug_title)
        #self.__pipeline__[

    next = __next__ # python 2

class AugDialog(QDialog):
    def __init__(self, config=None):
        # Config tells what noises are active, what the parameters are
        super(AugDialog, self).__init__()
        uic.loadUi('./src/qt_designer_file/dialogAug.ui', self)
        self.config = config
        self.__loadAugs__(augList)
        self.defaultImage = './src/data/tmp/car_detection_sample.png'
        self.__loadInitialImage__()
    
    def __loadAugs__(self, augList):
        for aug in augList.items():
            _item = QListWidgetItem()
            _item.setText(aug[0])
            _item.setCheckState(CheckState.Unchecked)
            self.listWidget.addItem(_item)
    
    def __loadInitialImage__(self):
        self._img = cv2.imread(self.defaultImage)
        h,w,_ = self._img.shape
        new_h = 500
        new_w = int((new_h/h)*w)
        self._img = cv2.resize(self._img, (new_w, new_h))

    def __loadExample__(self, ):
        # Assuming default image:
        _copy = np.copy(self._img)
        qtImage = images.convertCV2QT(_copy, 1000, 500)
        self.previewImage.setPixmap(qtImage)

    def __loadAugSelection__(self, aug):
        # change GUI:
        self.__loadExample__

    def __changeNoiseSelection__(self, target):
        return 0

    def __applyConfig__(self):
        # update config given
        return 0

# Augmentation holder:
mainAug = AugmentationPipeline(augList)


if __name__ == '__main__':
    img = cv2.imread('./data/samples/bus.jpg')
    out_img = gaussian_blur(np.copy(img), 10)
    cv2.imshow('test', out_img)
    cv2.waitKey(-1)
    while True:
        out_img2 = gaussian_blur(np.copy(img), (10, 90))
        cv2.imshow('test2', out_img2)
        cv2.waitKey(-1)
