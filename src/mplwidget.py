# ------------------------------------------------------
# -------------------- mplwidget.py --------------------
# ------------------------------------------------------
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvas

from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

class MplWidget(QWidget):
    
    def __init__(self, parent = None):
        super().__init__(parent)
        QWidget.__init__(self, parent)
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        
        self.vertical_layout = QVBoxLayout()
        self.vertical_layout.addWidget(self.canvas)
        
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(self.vertical_layout)

    


