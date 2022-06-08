import os
import urllib
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from time import sleep

# PyQt5
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QListWidgetItem
from PyQt5 import uic
from PyQt5.QtCore import QObject, QThread, pyqtSignal

# for weights in Google drive:
import gdown

class DownloadWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(float)
    logProgress = pyqtSignal(str)

    def __init__(self, pathDict:dict):
        super(DownloadWorker, self).__init__()
        self.pathDict = pathDict

    def run(self):
        self.checkWeightsExists(self.pathDict)
        self.finished.emit()
    
    def downloadMITWeight(self, filename:str):
        HOST_URL = "http://sceneparsing.csail.mit.edu/model/pytorch/"
        new_url = "%s/%s"%(HOST_URL, filename)
        self.logProgress.emit('Downloading MIT Segmentation: %s\n'%(filename))

        # get contents of directory:
        try:
            _content = requests.get(new_url).text
            p_content = BeautifulSoup(_content, 'html.parser')
            weight_files = [node.get('href') for node in p_content.find_all('a') if '.pt' in node.get('href')]
        except Exception as e:
            print("Error in parsing web directory. Make sure %s is available!"%(new_url))
            print(e)

        try:
            for _file in weight_files:
                self.progress.emit(0)
                
                def callback(blocknum, blocksize, totalsize):
                    readsofar = blocknum*blocksize
                    if totalsize > 0:
                        percent = readsofar / totalsize
                        self.progress.emit(percent)
                    else: # total size is unknown
                        self.progress.emit(0)

                file_url = "%s/%s"%(new_url, _file)
                _folder = './src/mit_semseg/ckpt/%s'%(filename)
                if not os.path.exists(_folder): os.mkdir(_folder)
                self.logProgress.emit("Downloading %s\n"%(file_url))
                request = urllib.request.urlretrieve(file_url, os.path.join(_folder, _file), callback)
            
        except Exception as e:
            print("Failed in grabbing needed files: %s"%(str(weight_files)))
            print(e)

    def downloadYOLOv3Weights(self, save_path, host='https://pjreddie.com/media/files/yolov3.weights'):
        self.progress.emit(0)
        self.logProgress.emit('Downloading YOLOv3 weights from %s\n'%(host))
        def callback(blocknum, blocksize, totalsize):
            readsofar = blocknum*blocksize
            if totalsize > 0:
                percent = readsofar / totalsize
                self.progress.emit(percent)
            else: # total size is unknown
                self.progress.emit(0)
        request = urllib.request.urlretrieve(host, save_path, callback)

    def downloadDETRWeights(self, save_path, host='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth'):
        self.progress.emit(0)
        self.logProgress.emit('Downloading DETR weights from %s\n'%(host))
        def callback(blocknum, blocksize, totalsize):
            readsofar = blocknum*blocksize
            if totalsize > 0:
                percent = readsofar / totalsize
                self.progress.emit(percent)
            else: # total size is unknown
                self.progress.emit(0)
        request = urllib.request.urlretrieve(host, save_path, callback)

    def downloadYOLOv4Weights(self, save_path, google_url='https://drive.google.com/file/d/1TSvLHH48eJJk7Glr5p2lscVet2jCazhi/view?usp=sharing'):
        self.progress.emit(0)
        self.logProgress.emit('Downloading YOLOv4 weights from %s\n'%(google_url))
        gdown.download(url=google_url, output=save_path, quiet=False, fuzzy=True)
        self.progress.emit(100.0)

    def downloadGoogleDriveWeights(self, save_path, google_url):
        self.progress.emit(0)
        self.logProgress.emit('Downloading YOLOv3 Face weights from %s\n'%(google_url))
        gdown.download(url=google_url, output=save_path, quiet=False, fuzzy=True)
        self.progress.emit(100.0)

    def checkWeightsExists(self, path_dict:dict):
        # path_dict: key=model_name; val=model_type
        print("Checking weights...")
        for path in path_dict.items():
            if path[0] == 'mit_semseg':
                self.progress.emit(0)
                _path_base = os.path.join('./src/mit_semseg/ckpt/', path[1])
                print(_path_base)
                if not os.path.exists('./src/mit_semseg/ckpt'): os.mkdir('./src/mit_semseg/ckpt')
                if not os.path.exists(_path_base):
                    print("MIT Segmentation weights (%s) not found. Attempting to download..."%(path[1]))
                    self.downloadMITWeight(path[1])
            elif path[0] == 'yolov3':
                self.progress.emit(0)
                _path_base = os.path.join('./src/obj_detector/weights', path[1])
                print(_path_base)
                if not os.path.exists('./src/obj_detector/weights'): os.mkdir('./src/obj_detector/weights')
                if not os.path.exists(_path_base):
                    print("YOLOv3 COCO weights not found. Attempting to download...")
                    self.downloadYOLOv3Weights(_path_base)
            elif path[0] == 'detr':
                self.progress.emit(0)
                _path_base = os.path.join('./src/detr/weights', path[1])
                print(_path_base)
                if not os.path.exists('./src/detr/weights'): os.mkdir('./src/detr/weights')
                if not os.path.exists(_path_base):
                    print("DETR COCO weights not found. Attempting to download...")
                    self.downloadDETRWeights(_path_base)
            elif path[0] == 'yolov4':
                self.progress.emit(0)
                _path_base = os.path.join('./src/yolov4/weights', path[1])
                print(_path_base)
                if not os.path.exists('./src/detr/weights'): os.mkdir('./src/detr/weights')
                if not os.path.exists(_path_base):
                    print("YOLOv4 COCO weights not found. Attempting to download...")
                    self.downloadYOLOv4Weights(_path_base)
            elif path[0] == 'yolov3-face':
                self.progress.emit(0)
                _path_base = os.path.join('./src/obj_detector/weights', path[1])
                print(_path_base)
                if not os.path.exists('./src/obj_detector/weights'): os.mkdir('./src/obj_detector/weights')
                if not os.path.exists(_path_base):
                    print("YOLOv3 COCO weights not found. Attempting to download...")
                    self.downloadGoogleDriveWeights(_path_base, "https://drive.google.com/u/0/uc?id=1OcabdJV98TaPg6D6LjWSNc6pl0s9IIdr")
            elif path[0] == 'yolox':
                self.progress.emit(0)
                _path_base = os.path.join('./src/yolox/weights', path[1])
                print(_path_base)
                if not os.path.exists('./src/yolox/weights'): os.mkdir('./src/yolox/weights')
                if not os.path.exists(_path_base):
                    print("YOLOvX COCO weights not found. Attempting to download...")
                    self.downloadYOLOv3Weights(_path_base, host='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth')

        self.logProgress.emit("Finished downloading all weights. Press Continue to proceed to main GUI\n")

# Setup Dialog Window:
class Downloader(QDialog):
    def __init__(self, pathDict:dict):
        super(Downloader, self).__init__()
        uic.loadUi('./src/qt_designer_file/weightDownloader.ui', self)
        self.logText.setEnabled(False)
        self.progressBar.setValue(0)
        self.switchStates(0)
        self.pathDict = pathDict
        self.afterDownload.setEnabled(False)
        self.downloadButton.clicked.connect(self.__startDownload__)
        self.afterDownload.clicked.connect(self.close)

    def switchStates(self, _type:int):
        _type = bool(_type)
        # hide the progress stuff:
        self.logText.setVisible(_type)
        self.progressBar.setVisible(_type)
        self.afterDownload.setVisible(_type)
        self.detailExplain.setVisible(not _type)
        self.titleExplain.setVisible(not _type)
        self.downloadButton.setVisible(not _type)

    def __startDownload__(self):
        self.switchStates(1)
        self.logText.insertPlainText('Checking weights...\n')
        #self.checkWeightsExists(self.pathDict)
        self.runThread()

    def runThread(self):
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = DownloadWorker(self.pathDict)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)
        self.worker.logProgress.connect(self.updateLog)
        # Step 6: Start the thread
        self.thread.start()

        # Final resets
        
        self.thread.finished.connect(
            lambda: self.afterDownload.setEnabled(True)
        )

    def reportProgress(self, val):
        self.progressBar.setValue(val*100)

    def updateLog(self, val):
        self.logText.insertPlainText(val)

    @staticmethod
    def check(path_dict:dict):
        for path in path_dict.items():
            if path[0] == 'mit_semseg':
                _path_base = os.path.join('./src/mit_semseg/ckpt/', path[1])
                if not os.path.exists(_path_base):
                    return True
            elif path[0] == 'yolov3':
                _path_base = os.path.join('./src/obj_detector/weights', path[1])
                if not os.path.exists(_path_base):
                    return True
            elif path[0] == 'detr':
                _path_base = os.path.join('./src/detr/weights', path[1])
                if not os.path.exists(_path_base):
                    return True
            elif path[0] == 'yolov4':
                _path_base = os.path.join('./src/yolov4/weights', path[1])
                if not os.path.exists(_path_base):
                    return True
            elif path[0] == 'yolov3-face':
                _path_base = os.path.join('./src/obj_detector/weights', path[1])
                if not os.path.exists(_path_base):
                    return True
            elif path[0] == 'yolox':
                _path_base = os.path.join('./src/yolox/weights', path[1])
                if not os.path.exists(_path_base):
                    return True
        return False