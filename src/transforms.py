import random
import math
import os
from urllib import request
from PyQt5.QtCore import QObject, pyqtSignal, Qt

from numpy.lib.function_base import select
from src.utils.qt5extra import CheckState
#import utils.qt5extra.CheckState

import PyQt5
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QListWidgetItem, QMessageBox, QWidget
from PyQt5 import uic
import cv2
import numpy as np
import time
from src.utils import images
#import src.utils.images

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
    """
    Dims the intensity of the image by the give factor/range of factor. 
    
        |Parameters: 
            |image (numpy array): The original input image
            |factor (float or tuple): The diming factor (if float) or the dimimg factor range (if tuple)
            |seed (int or 1-d array_like): Seed for RandomState. Must be convertible to 32 bit unsigned integers.
        
        |Returns: 
            |image (numpy array): The dimed image  
    """
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
    else: assert False

def gaussian_blur(image, kernel_size_factor, stdX=0, stdY=0, seed=-1):
    if type(kernel_size_factor) == float or type(kernel_size_factor) == int:
        w = int((kernel_size_factor*2)+1)
        h = int((kernel_size_factor*2)+1)
        blur_img = cv2.GaussianBlur(
            image, (w, h), cv2.BORDER_DEFAULT, stdX, stdY)
        return blur_img
    if type(kernel_size_factor) == tuple:
        if seed != -1:
            np.random.seed(seed)
        lower, upper = kernel_size_factor
        random_size_factor = np.random.uniform(lower, upper)
        w_r = (int(random_size_factor)*2)+1
        h_r = (int(random_size_factor)*2)+1
        blur_img = cv2.GaussianBlur(
            image, (w_r, h_r), cv2.BORDER_DEFAULT, stdX, stdY)
        return blur_img

def jpeg_comp(image, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, enc_img = cv2.imencode('.jpg', image, encode_param)
    if result is True:
        dec_img = cv2.imdecode(enc_img, 1)
        return dec_img

def normal_comp(image, scale_factor):
    original_shape = image.shape
    width = int(image.shape[1] * scale_factor / 100)
    height = int(image.shape[0] * scale_factor / 100)
    dim = (width, height)
    # resize image
    resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    resized_img = cv2.resize(resized_img, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC) 
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

def flipAxis(image, mode):
    if mode > 0:
        return cv2.flip(image, 1) # Flips along vertical axis
    elif mode == 0:
        return cv2.flip(image, 0) # Flips along horizontal axis
    else: 
        return cv2.flip(image, -1) # Flips along both axes

def flipVertical(image):
    image = cv2.flip(image, 0)
    return image
    
def fisheye(image, factor=0.25):
    '''
    Transform image using fisheye projection
    |Parameters: 
        |image (numpy array): The original input image
        |center (array): [x, y] values of the center of transformation
        |factor (float): The distortion factor for fisheye effect
    |Returns: 
        |image (numpy array): The transformed image 
    '''
    new_image = np.zeros_like(image)
    width, height = image.shape[0], image.shape[1]
    w, h = float(width), float(height)
    for x in range(len(new_image)):
        for y in range(len(new_image[x])):
            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)
            # get xn and yn euclidean distance from normalized center
            radius = np.sqrt(xnd**2 + ynd**2)
            # get new normalized pixel coordinates
            if 1 - factor*(radius**2) == 0:
                new_xnd, new_ynd = xnd, ynd
            else:
                new_xnd = xnd / (1 - (factor*(radius**2)))
                new_ynd = ynd / (1 - (factor*(radius**2)))
            # convert the new normalized distorted x and y back to image pixels
            new_x, new_y = int(((new_xnd + 1)*w)/2), int(((new_ynd + 1)*h)/2)
            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= new_x and new_x < width and 0 <= new_y and new_y < height:
                new_image[x][y] = image[new_x][new_y]
    return new_image

def barrel(image, factor=0.005):
    height, width, channel = image.shape
    k1, k2, p1, p2 = factor, 0, 0, 0
    dist_coeff = np.array([[k1],[k2],[p1],[p2]])
    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)
    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 10.        # define focal length x
    cam[1,1] = 10.        # define focal length y
    new_image = cv2.undistort(image, cam, dist_coeff)
    return new_image

def pick_img(start_dir):
    curr_dir = os.listdir(os.path.join(start_dir))
    # curr_dir.remove("LABELS")
    curr_path = start_dir
    
    while True:
        curr_file = random.choice(curr_dir)

        if os.path.isfile(os.path.join(curr_path, curr_file)):
            img = cv2.imread(os.path.join(curr_path, curr_file))
            if img is None:
                curr_dir = os.listdir(os.path.join(start_dir))
                # curr_dir.remove("LABELS")
                curr_path = start_dir
            else:
                return img
        else:
            curr_path = os.path.join(curr_path, curr_file)
            curr_dir = os.listdir(os.path.join(curr_path))

def simple_mosaic(image, dummy):
    # pick three images
    images = [pick_img('imgs') for x in range(4)]
    # images += [image]

    # find smallest image, resize others to fit
    smallest = image.shape[0] * image.shape[1]
    sm_shape = image.shape
    for i in images:
        curr_area = i.shape[0] * i.shape[1]
        if curr_area < smallest:
            smallest = curr_area
            sm_shape = i.shape

    # combine images into one big 2x2
    resized = [cv2.resize(curr_im, (sm_shape[0], sm_shape[1])) for curr_im in images]
    big_image = []
    big_image = np.concatenate((resized[0], resized[1]), axis=1)
    bottom = np.concatenate((resized[2], resized[3]), axis=1)
    big_image = np.concatenate((big_image, bottom), axis=0)
    
    # pick random bounds to make the mosaic image
    row_start = math.floor(random.random() * big_image.shape[0] / 2)
    col_start = math.floor(random.random() * big_image.shape[1] / 2)
    row_end = row_start + math.floor(big_image.shape[0] / 2)
    col_end = col_start + math.floor(big_image.shape[1] / 2)
    final_im = big_image[row_start:row_end][col_start:col_end]
    return final_im

def black_white(image, channel):
    channel = int(channel)
    image[:,:,0] = image[:,:,channel]
    image[:,:,1] = image[:,:,channel]
    image[:,:,2] = image[:,:,channel]
    return image

def speckle_noise(image, dummy):
    '''
    Speckle is a granular noise that inherently exists in an image and degrades its quality. 
    It can be generated by multiplying random pixel values with different pixels of an image.
    '''
    gauss = np.random.normal(0,1,image.size)
    gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')
    noise = image + image * gauss
    return noise

def saturation (image, factor=50):
    '''
    Saturation impacts the color intensity of the image, making it more vivid or muted depending
    on the value.
    '''
    
    hsvimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    
    (h, s, v) = cv2.split(hsvimg)
    fac = 6.5025
    
    s[:] = s * ((factor/100) * fac)

    s = np.clip(s,0,255)
    imghsv = cv2.merge([h,s,v])
    
    img_sated = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return img_sated

def alternate_mosaic(image, num_slices):
    if num_slices == 1: return image
    width, height = image.shape[0], image.shape[1]
    new_image = np.zeros_like(image)
    
    x_size = int(width/num_slices)
    while(width % x_size != 0):
        width -= 1
        x_size = int(width/num_slices)
    y_size = int(height/num_slices)
    while(height % y_size != 0):
        height -= 1
        y_size = int(height/num_slices)
    mats = []
    x,y = 0,0
    while x < width:
        y = 0
        while y < height:
            app = image[x:x+x_size,y:y+y_size,:]
            if len(mats) != 0:
                if app.shape != mats[0].shape: break
            mats.append(app)
            y += y_size
        x += x_size
    random.shuffle(mats)
    x,y = 0,0
    i = 0
    while x < width:
        y = 0
        while y < height:
            if i == len(mats):break
            new_image[x:x+x_size,y:y+y_size,:] = mats[i]
            y += y_size
            i += 1
        x += x_size
    return new_image

augList = {
    "Intensity": {"function": dim_intensity, "default": [0.5], "example":0.5},
    "Gaussian Noise": {"function": gaussian_noise, "default": [1,25,50], "example":25},
    "Gaussian Blur": {"function": gaussian_blur, "default": [30], "example":30},
    "JPEG Compression": {"function": jpeg_comp, "default": [100,75,50], "example":20},
    "Normal Compression": {"function": normal_comp, "default": [20], "example":30},
    "Salt and Pepper": {"function": saltAndPapper_noise, "default": [0.01, 0.2, 0.3], "example":0.25},
    "Flip Axis": {"function": flipAxis, "default": [-1], "example": -1},
    "Fisheye": {"function": fisheye, "default": [0.2, 0.3, 0.4], "example":0.4},
    "Barrel": {"function": barrel, "default": [0.05, 0.005, 0.0005], "example":0.005},
    "Simple Mosaic": {"function": simple_mosaic, "default":[], "example":[]},
    "Black and White": {"function": black_white, "default":[0,1,2], "example":0}, 
    "Speckle Noise": {"function": speckle_noise, "default":[], "example":[]},
    "Saturation" : {"function": saturation, "default":[50], "example":50},
    "Alternate Mosaic": {"function": alternate_mosaic, "default":[1,2,3,4,5], "example":2} # 1x1 - 5x5
}

class Augmentation:
    """
    Creates and Add Augmentations
    """
    def __init__(self, aug, original_position, args, verbose=True) -> None:
        self.__title__ = aug[0]
        self.__run__ = aug[1]
        self.__checked__ = False
        self.__position__ = original_position
        self.__args__ = args[0] # list of values
        self.__example__ = args[1]
        self.__verbose__ = verbose

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
    def args(self):
        return self.__args__

    @property
    def exampleParam(self):
        return self.__example__

    def setExampleParam(self, value):
        self.__example__ = value

    def __call__(self, image, request_param=None, example=False):
        if example:
            _param = [self.__example__]
            if self.__verbose__:
                if example and not request_param is None:
                    print("WARNING: Request param is ignored since example is set")
        else:
            if not request_param is None:
                _param = [request_param]
                if not request_param in self.__args__:
                    if self.__verbose__:
                        print("WARNING: Requested params not in set arguments. Set verbose to false to dismiss")
            else:
                if self.__verbose__: print("WARNING: No request given. Example is false, so returning example value")
                _param = [self.__example__] 
        return self.__run__(image, *_param)

    def setParam(self, args):
        self.__args__ = args

class AugmentationPipeline():
    def __init__(self, augList:dict) -> None:
        self.__list__ = augList
        self.__keys__ = list(self.__list__.keys())
        self.__augList__ = []
        self.__index__ = 0
        self.__pipeline__ = []
        self.__wrapper__()

    def __wrapper__(self):
        for pos, item in enumerate(self.__list__.items()):
            _item = Augmentation( (item[0], item[1]["function"]), pos, args=(item[1]["default"], item[1]["example"]), verbose=False, )
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
        # make more efficient later:
        for item in self.__pipeline__:
            if title == item.title:
                return True
        return False

    def index(self, title):
        for i, item in enumerate(self.__pipeline__):
            if title == item.title:
                return i
        return -1

    def append(self, aug_title, param=None, example=None):
        augIndex = self.__keys__.index(aug_title)
        augItem = self.__augList__[augIndex]
        if not param is None: 
            augItem.setParam(param)
        if not example is None: augItem.setExampleParam(example)
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
        with open(filename, 'r') as f:
            content = list(map(str.strip, f.readlines()))
        self.clear()

        # format: [title,# of parameters,*parameters,1,example]
        for _content in content:
            _content = _content.split(',')
            name = _content[0]
            nargs = int(_content[1])
            params = []
            for i in range(nargs):
                params.append( float(_content[i+2]) )
            params = list(params)
            _example_buffer_loc = nargs+3 #+2 and +1 to get to 1
            example = float(_content[_example_buffer_loc])

            if name in augList:
                _aug  = Augmentation([name, augList[name]['function']], list(augList.keys()).index(name), [params, example], verbose=False)
                mainAug.__pipeline__.append(_aug)
            else:
                print("Augmentation name is not recognized! Ignoring this line")

    def save(self, filename):
        if '.txt' not in filename[0]:
            _filename = "%s.txt"%(filename[0])
        else:
            _filename = filename[0]
        
        with open(_filename, 'w') as f:
            for aug in self.__pipeline__:
                aug_title = aug.__title__
                parameters = [str(i) for i in aug.__args__]
                para_out = ','.join(parameters)
                para_length = len(parameters)
                str_out = "%s,%i,%s,1,%f\n"%(aug_title, para_length, para_out,aug.exampleParam)
                f.write(str_out)

    def checkArgs(self):
        maxLen = 0
        for aug in self.__pipeline__:
            if maxLen == 0:
                maxLen = len(aug.args)
            if maxLen != len(aug.args):
                return False, "Compounding augmentations require equal number of parameters for each active parameter. %s has mismatch of %i parameters"%(aug.title, len(aug.args))
        return True, ""

    next = __next__ # python 2

class AugDialog(QDialog):
    pipelineChanged = pyqtSignal(object)

    def __init__(self, listViewer):
        # Config tells what noises are active, what the parameters are
        super(AugDialog, self).__init__()
        self.__viewer__ = listViewer # outside of the Augmentation Dialog UI
        self.lastRow = 0
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
        _btn2.clicked.connect(self.close)

    def __loadEvents__(self):
        self.listWidget.itemClicked.connect(self.__loadAugSelection__)

    def __loadAugs__(self):
        for aug in mainAug.__augList__:
            _item = QListWidgetItem()
            _item.setText(aug.title)
            _item.setCheckState(CheckState.Unchecked)
            _item.setData(Qt.UserRole, [aug, "", ""]) # aug, parameters, example
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
        self.listWidget.setCurrentItem(self.listWidget.itemAt(0,0))

    def __loadAugSelection__(self, aug):
        # update old active aug:
        _payload = self.listWidget.item(self.lastRow).data(Qt.UserRole)
        _payload[1] = self.noiseRange.text()
        _payload[2] = self.exampleLine.text()
        self.listWidget.item(self.lastRow).setData(Qt.UserRole, _payload)
        
        # change GUI when item is clicked
        currentItem = aug.text()
        _payload = aug.data(Qt.UserRole)
        augIndex = mainAug.__keys__.index(currentItem)
        augItem = mainAug.__augList__[augIndex]
        
        if _payload[1] == '': 
            strArgs = [ str(i) for i in augItem.args]
            parameters = ",".join(strArgs)
        else: parameters = _payload[1]
        
        if _payload[2] == '':
            example = augItem.exampleParam
        else: example = _payload[2]

        # GUI range controls:
        self.noiseRange.setText(parameters)
        self.exampleLine.setText(str(example))

        _copy = np.copy(self._img)
        _copy = augItem(_copy, example=True)
        qtImage = images.convertCV2QT(_copy, 1000, 500)
        self.previewImage.setPixmap(qtImage)
        self.lastRow = augIndex

    # change GUI to match mainAug
    def __applyConfig__(self):
        # update config given:
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

    def closeEvent(self, event):
        for i in range(self.listWidget.count()):
            listItem = self.listWidget.item(i)
            payload = listItem.data(Qt.UserRole)
            payload[1] = ''; payload[2] = ''
            self.listWidget.item(i).setData(Qt.UserRole, payload)
        event.accept()

    # change mainAug to match selected items from GUI:
    def __applySelection__(self):
        # update the active item:
        cr = self.listWidget.currentRow()
        _payload = self.listWidget.item(cr).data(Qt.UserRole)
        _payload[1] = self.noiseRange.text()
        _payload[2] = self.exampleLine.text()
        self.listWidget.item(cr).setData(Qt.UserRole, _payload)

        # get checks from listWidget:
        for i in range(self.listWidget.count()):
            listItem = self.listWidget.item(i)

            # parse the list items:
            _payload = listItem.data(Qt.UserRole)
            _noiseRange = _payload[1]
            _example = _payload[2]

            if _noiseRange != '':
                try: _param = [float(i) for i in _noiseRange.split(',')]
                except Exception as e: print("Failed to convert string to array of floats")
                # probably do a signal here to update the UI
            else: _param = None

            if _example != '': 
                try:
                    _example = float(_payload[-1])
                except Exception as e: print("Failed to convert example to number")
            else: _example = None

            itemIndex = mainAug.index(listItem.text())

            if listItem.checkState() and itemIndex == -1:
                mainAug.append(listItem.text(), param=_param, example=_example)
            elif listItem.checkState() and itemIndex != -1:
                if not _param is None: mainAug.__pipeline__[itemIndex].setParam(_param)
                if not _example is None: mainAug.__pipeline__[itemIndex].setExampleParam(_example)
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
        if selected_idx != -1:
            if selected_idx != len(mainAug)-1:
                #print("running!")
                item = self.__viewer__.takeItem(selected_idx)
                #print(item)
                #self.__viewer__.removeItemWidget(item)
                mainAug.__pipeline__.insert(selected_idx+1, mainAug.__pipeline__.pop(selected_idx))

                self.__viewer__.insertItem(selected_idx+1, item)
                self.__updateViewer__()
                self.__viewer__.setCurrentRow(selected_idx+1)

    def __moveUp__(self):
        selected_idx = self.__viewer__.currentRow()
        if selected_idx != -1:
            if selected_idx != 0:
                #print("running!")
                item = self.__viewer__.takeItem(selected_idx)
                #print(item)
                #self.__viewer__.removeItemWidget(item)
                mainAug.__pipeline__.insert(selected_idx-1, mainAug.__pipeline__.pop(selected_idx))

                self.__viewer__.insertItem(selected_idx-1, item)
                self.__updateViewer__()
                self.__viewer__.setCurrentRow(selected_idx-1)

    def demoAug(self):
        mainAug.clear()
        mainAug.append('Gaussian Noise')
        mainAug.append('JPEG Compression')
        mainAug.append('Salt and Pepper')
        self.__updateViewer__()

# Augmentation holder:
mainAug = AugmentationPipeline(augList)