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

class DownloadWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(float)
    logProgress = pyqtSignal(str)

    def __init__(self, pathDict:dict):
        super(DownloadWorker, self).__init__()
        self.pathDict = pathDict

    def run(self):
        """Long-running task."""
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
                file_url = "%s/%s"%(new_url, _file)
                _folder = './src/ckpt/%s'%(filename)
                if not os.path.exists(_folder): os.mkdir(_folder)
                response = requests.get(file_url)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                blocksize = 32768
                current_bytes_read = 0
                with open(os.path.join(_folder, _file), "wb") as handle:
                    for data in response.iter_content():
                        handle.write(data)
                        current_bytes_read += len(data)
                        percent = current_bytes_read/total_size_in_bytes
                        #print(percent)
                        self.progress.emit(percent)
                        #self.progressBar.setValue(percent*100)
        except Exception as e:
            print("Failed in grabbing needed files: %s"%(str(weight_files)))
            print(e)

    def downloadYOLOv3Weights(self, save_path, host='https://pjreddie.com/media/files/yolov3.weights'):
        self.logProgress.emit('Downloading YOLOv3 COCO from pjreddie\n')

        response = requests.get(host)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        blocksize = 32768
        current_bytes_read = 0
        print(total_size_in_bytes, host)
        self.logProgress.emit('Starting download of YOLOv3 weights from %s\n'%(host))
        with open(save_path, "wb") as handle:
            for data in response.iter_content(blocksize):
                handle.write(data)
                current_bytes_read += len(data)
                percent = current_bytes_read/total_size_in_bytes
                self.progress.emit(percent)
                #self.progressBar.setValue(percent*100)

    def checkWeightsExists(self, path_dict:dict):
        # path_dict: key=model_name; val=model_type
        print("Checking weights...")
        for path in path_dict.items():
            if path[0] == 'mit_semseg':
                self.progress.emit(0)
                _path_base = os.path.join('./src/ckpt/', path[1])
                print(_path_base)
                if not os.path.exists('./src/ckpt'): os.mkdir('./src/ckpt')
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
        '''
        self.thread.finished.connect(
            lambda: self.longRunningBtn.setEnabled(True)
        )
        '''

    def reportProgress(self, val):
        self.progressBar.setValue(val)

    def updateLog(self, val):
        self.logText.insertPlainText(val)

#if __name__ == '__main__':
#    downloadMITWeight("ade20k-resnet18dilated-c1_deepsup")