import random
from PyQt5.QtCore import QObject, pyqtSignal

from PyQt5.uic.uiparser import QtCore
from numpy.lib.function_base import select
from src.utils.qt5extra import CheckState
import PyQt5
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QListWidgetItem
from PyQt5 import uic
import cv2
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
        w = int((kernel_size_factor*2)+1)
        h = int((kernel_size_factor*2)+1)
        print(w,h)
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
    original_shape = image.shape
    width = int(image.shape[1] * scale_factor / 100)
    height = int(image.shape[0] * scale_factor / 100)
    dim = (width, height)
    # resize image
    resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    resized_img = cv2.resize(resized_img, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC) 
    print('Resized Dimensions : ', resized_img.shape)
    return resized_img

def saltAndPapper_noise(image, prob=0.01):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    #image = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(image.shape[:2])
    image[probs < (prob / 2)] = black
    image[probs > 1 - (prob / 2)] = white
    return image

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
    #"Poisson Noise": poisson_noise,
    #"Speckle Noise": speckle_noise,
}

augDefaultParams = {
    "Intensity": [0.0,0.5], #ranges
    "Gaussian Noise": [0,50],
    "Gaussian Blur": [10,30],
    "JPEG Compression": [5],
    "Normal Compression": [20],
    "Salt and Pepper": [0.05, 0.3],
    #"Poisson Noise": [],
    #"Speckle Noise": [],
}

assert len(list(augList.keys())) == len(list(augDefaultParams.keys())), "Default parameters are not the same length as augmentation list. If no default values, leave an empty list"

class Augmentation:
    def __init__(self, aug, original_position, *args) -> None:
        self.__title__ = aug[0]
        self.__run__ = aug[1]
        self.__checked__ = False
        self.__position__ = original_position
        self.__function_args__ = args # range of values
        if len(self.__function_args__) == 2: 
            low, high = self.__function_args__
            self.__example__ = (low+high)/2
        elif len(self.__function_args__) == 1:
            self.__example__ = self.__function_args__[0]
        else:
            self.__example__ = 0

    @property
    def title(self):
        return self.__title__

    @property
    def enabled(self):
        return self.__checked__

    @property
    def position(self):
        return self.__position__
    
    @property
    def function_arg(self):
        return self.__function_args__

    @property
    def exampleParam(self):
        return self.__example__

    @property
    def setExampleParam(self, value):
        self.__example__ = value

    def __call__(self, image, example=False, dtype=float):
        # random between function arg ranges:
        if len(self.__function_args__) == 1:
            _param = self.__function_args__
        elif example:
            _param = self.__example__
            if not isinstance(_param, list): 
                _param = [_param]
        elif len(self.__function_args__) == 2:
            if dtype == float:
                _param = random.uniform(*self.__function_args__)
            else:
                _param = random.randint(*self.__function_args__)
            _param = (_param)
        else:
            print("WARNING: Passing no parameters. Assuming 0...")
            _param = self.__example__

        print(_param)
        return self.__run__(image, *_param)

    def setParam(self, *args):
        self.__function_args__ = args

class AugmentationPipeline():
    def __init__(self, augList:dict, defaultParams:dict) -> None:
        self.__list__ = augList
        self.__keys__ = list(self.__list__.keys())
        self.__defaultParams__ = defaultParams
        self.__augList__ = []
        self.__index__ = 0
        self.__pipeline__ = []
        self.__wrapper__()

    def __wrapper__(self):
        _default_items = [item[1] for item in self.__defaultParams__.items()]
        for pos, item in enumerate(self.__list__.items()):
            _item = Augmentation(item, pos, *_default_items[pos])
            self.__augList__.append(_item)

    def __len__(self):
        return len(self.__pipeline__)

    def __iter__(self):
        return (self.__pipeline__[x] for x in range(len(self.__pipeline__)))

    def __getitem__(self, key):
        self.__pipeline__[key]

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
            _out += '%s - %s\n'%(pipe.title, pipe.position)
        return _out

    def exists(self, title):
        for item in self.__pipeline__:
            if title == item.title:
                return True
        return False

    def append(self, aug_title):
        augIndex = self.__keys__.index(aug_title)
        augItem = self.__augList__[augIndex]
        self.__pipeline__.append(augItem)

    def remove(self, aug_title):
        augIndex = self.__keys__.index(aug_title)
        for i, aug in enumerate(self.__pipeline__):
            if aug.position == augIndex:
                self.__pipeline__.remove(aug)
                break

    def clear(self):
        self.__pipeline__.clear()
        self.__index__ = 0
        return 0

    def load(self, filename):
        # check if filename is a .txt:
        print(filename)
        with open(filename, 'r') as f:
            content = list(map(str.strip, f.readlines()))

        # format: [title,# of parameters,*parameters]
        for _content in content:
            _content = _content.split(',')
            name = _content[0]
            nargs = int(_content[1])
            params = []
            for i in range(nargs):
                params.append( float(_content[i+2]) )
            
            if name in augList:
                _aug  = Augmentation([name, augList[name]], list(augList.keys()).index(name), params)
                mainAug.__pipeline__.append(_aug)
            else:
                print("Augmentation name is not recognized! Ignoring this line")
        print(mainAug.__pipeline__)

    def save(self, filename):
        print(filename)
        if '.txt' not in filename[0]:
            _filename = "%s.txt"%(filename[0])
        else:
            _filename = filename[0]
        
        with open(_filename, 'w') as f:
            for aug in self.__pipeline__:
                aug_title = aug.__title__
                parameters = [str(i) for i in aug.__function_args__]
                para_out = ','.join(parameters)
                para_length = len(parameters)
                str_out = "%s,%i,%s\n"%(aug_title, para_length, para_out)
                f.write(str_out)

    next = __next__ # python 2

class AugDialog(QDialog):
    pipelineChanged = pyqtSignal(object)

    def __init__(self, listViewer):
        # Config tells what noises are active, what the parameters are
        super(AugDialog, self).__init__()
        self.__viewer__ = listViewer # outside of the Augmentation Dialog UI
        uic.loadUi('./src/qt_designer_file/dialogAug.ui', self)
        self.__loadAugs__()
        self.__loadEvents__()
        self.defaultImage = './src/data/tmp/car_detection_sample.png'
        self.__loadInitialImage__()
        self.__loadExample__()
        self.savedAugPath = './src/data/saved_augs'
        self.__applyConfig__()
        _btn1, _btn2 = self.buttonBox.buttons() # ok, cancel
        _btn1.clicked.connect(self.__applySelection__)

    def __loadEvents__(self):
        self.listWidget.itemClicked.connect(self.__loadAugSelection__)

    def __loadAugs__(self):
        for aug in mainAug.__augList__:
            _item = QListWidgetItem()
            _item.setText(aug.title)
            _item.setCheckState(CheckState.Unchecked)
            self.listWidget.addItem(_item)
    
    def __loadInitialImage__(self):
        self._img = cv2.imread(self.defaultImage)
        h,w,_ = self._img.shape
        new_h = 500
        new_w = int((new_h/h)*w)
        self._img = cv2.resize(self._img, (new_w, new_h))

    def __loadExample__(self):
        # Assuming default image:
        _copy = np.copy(self._img)
        qtImage = images.convertCV2QT(_copy, 1000, 500)
        self.previewImage.setPixmap(qtImage)
        self.__loadAugSelection__(self.listWidget.itemAt(0,0))

    def __loadAugSelection__(self, aug):
        # change GUI when item is clicked
        currentItem = aug.text()
        augIndex = mainAug.__keys__.index(currentItem)
        augItem = mainAug.__augList__[augIndex]
        print(augItem.function_arg)
        if len(augItem.function_arg) == 2:
            low, high = augItem.function_arg #values
            example = augItem.exampleParam
            self.lowLine.setText(str(low))
            self.highLine.setText(str(high))
            self.exampleLine.setText(str(example))
            _copy = np.copy(self._img)
            _copy = augItem.__run__(_copy, augItem.exampleParam)
            qtImage = images.convertCV2QT(_copy, 1000, 500)
            self.previewImage.setPixmap(qtImage)
        elif len(augItem.function_arg) == 1:
            example = augItem.exampleParam
            self.exampleLine.setText(str(example))
            _copy = np.copy(self._img)
            _copy = augItem.__run__(_copy, augItem.exampleParam)
            qtImage = images.convertCV2QT(_copy, 1000, 500)
            self.previewImage.setPixmap(qtImage)
        else:
            _copy = np.copy(self._img)
            qtImage = images.convertCV2QT(_copy, 1000, 500)
            self.previewImage.setPixmap(qtImage)
            

    def __changeNoiseSelection__(self, target:Augmentation):
        _low = self.lowLine.text()
        _high = self.highLine.text()
        _example = self.exampleLine.text()
        _entry = []
        if _low != '':
            _entry.append(float(_low))
        if _high != '':
            _entry.append(float(_low))
        if _example != '':
            _example_value = float(_example)
            target.setExampleParam(_example_value)
        target.setParam(*_entry)

    # change GUI to match mainAug
    def __applyConfig__(self):
        # update config given:
        if len(mainAug) == 0:
            for i in range(self.listWidget.count()):
                listItem = self.listWidget.item(i)
                listItem.setCheckState(CheckState.Unchecked)
        else:
            for aug in mainAug:
                itemPos = aug.position
                listItem = self.listWidget.item(itemPos)
                listItem.setCheckState(CheckState.Checked)

    def show(self):
        self.__applyConfig__()
        return super().show()

    # change mainAug to match selected items from GUI:
    def __applySelection__(self):
        # get checks from listWidget:
        for i in range(self.listWidget.count()):
            listItem = self.listWidget.item(i)
            if(listItem.checkState() and not mainAug.exists(listItem.text())):
                mainAug.append(listItem.text())
            elif not listItem.checkState(): # make more efficient later
                for item in mainAug.__pipeline__:
                    if item.title == listItem.text():
                        mainAug.remove(listItem.text())
                        break
        self.__updateViewer__()
    
    def __updateViewer__(self):
        # add listviewer:
        self.__viewer__.clear()
        for item in mainAug:
            self.__viewer__.addItem(item.title)
        self.pipelineChanged.emit(None)

    def __loadFileDialog__(self):
        _file = QFileDialog.getOpenFileName(self, "Load in Augmentation", self.savedAugPath, '*.txt')
        if _file[0] != '':
            mainAug.load(_file[0])
            self.__applyConfig__() # change GUI
            self.__updateViewer__()

    def __saveFileDialog__(self):
        save_path = QFileDialog.getSaveFileName(self, 'Save Current Augmentation', self.savedAugPath, '*.txt')
        mainAug.save(save_path)

    def __deleteItem__(self):
        selected_item = self.__viewer__.currentItem()
        if selected_item is not None:
            for item in mainAug:
                if item.title == selected_item.text():
                    mainAug.remove(item.title)
                    self.__updateViewer__()
                    return 0

    def __moveDown__(self):
        selected_idx = self.__viewer__.currentRow()
        print(selected_idx)
        if selected_idx != -1:
            if selected_idx != len(mainAug)-1:
                #print("running!")
                item = self.__viewer__.takeItem(selected_idx)
                #print(item)
                #self.__viewer__.removeItemWidget(item)
                mainAug.__pipeline__.insert(selected_idx+1, mainAug.__pipeline__.pop(selected_idx))

                self.__viewer__.insertItem(selected_idx+1, item)
                self.__viewer__.setCurrentRow(selected_idx+1)
                #self.__updateViewer__()

    def __moveUp__(self):
        selected_idx = self.__viewer__.currentRow()
        print(selected_idx)
        if selected_idx != -1:
            if selected_idx != 0:
                #print("running!")
                item = self.__viewer__.takeItem(selected_idx)
                #print(item)
                #self.__viewer__.removeItemWidget(item)
                mainAug.__pipeline__.insert(selected_idx-1, mainAug.__pipeline__.pop(selected_idx))

                self.__viewer__.insertItem(selected_idx-1, item)
                self.__viewer__.setCurrentRow(selected_idx-1)
                #self.__updateViewer__()

    def demoAug(self):
        mainAug.clear()
        mainAug.append('Gaussian Noise')
        mainAug.append('JPEG Compression')
        mainAug.append('Salt and Pepper')
        self.__updateViewer__()

# Augmentation holder:
mainAug = AugmentationPipeline(augList, augDefaultParams)
print(mainAug)

if __name__ == '__main__':
    img = cv2.imread('./data/samples/bus.jpg')
    out_img = gaussian_blur(np.copy(img), 10)
    cv2.imshow('test', out_img)
    cv2.waitKey(-1)
    while True:
        out_img2 = gaussian_blur(np.copy(img), (10, 90))
        cv2.imshow('test2', out_img2)
        cv2.waitKey(-1)
