from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import QLabel, QWidget
from cv2 import imread

class Label(QLabel):

    def __init__(self, QWidget):
        super(Label, self).__init__()
        self.pixmap_width: int = 1
        self.pixmapHeight: int = 1

        self.img = None

        self.setAcceptDrops(True)


    def setPixmap(self, pm: QPixmap) -> None:
        self.pixmap_width = pm.width()
        self.pixmapHeight = pm.height()

        super(Label, self).setPixmap(pm)
        self.updateMargins()


    def resizeEvent(self, a0: QResizeEvent) -> None:
        self.updateMargins()
        super(Label, self).resizeEvent(a0)


    def updateMargins(self):
        if self.pixmap() is None:
            return
        pixmapWidth = self.pixmap().width()
        pixmapHeight = self.pixmap().height()
        if pixmapWidth <= 0 or pixmapHeight <= 0:
            return
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        if w * pixmapHeight > h * pixmapWidth:
            m = int((w - (pixmapWidth * h / pixmapHeight)) / 2)
            self.setContentsMargins(m, 0, m, 0)
        else:
            m = int((h - (pixmapHeight * w / pixmapWidth)) / 2)
            self.setContentsMargins(0, m, 0, m)

    
    def dragEnterEvent(self, e):
        #if e.mimeData().hasFormat('text/plain'):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()


    def dropEvent(self, e):
        print(e.mimeData().urls()[0].toLocalFile())
        #self.setText(e.mimeData().text())
        filepath = e.mimeData().urls()[0].toLocalFile()
        
        self.setPixmap(QPixmap(filepath))

        self.img = imread(filepath)

    
    def addImg(self, img):
        self.img = img

    
    def getImg(self):
        return self.img
