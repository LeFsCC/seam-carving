import sys

import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QRadioButton, QFileDialog, QProgressBar, QDoubleSpinBox, \
    QButtonGroup
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import QPoint, QDir, QBasicTimer, QThread
from PyQt5 import QtCore
import numpy as np
import seamCarving


class MyThread(QThread):
    def __init__(self, seam_carving, raw_image, scale, dire, energy):
        self.seam_carving = seam_carving
        self.raw_image = raw_image
        self.scale = scale
        self.dire = dire
        self.energy = energy
        super(MyThread, self).__init__()

    def run(self):
        self.seam_carving.carve(self.raw_image, self.scale, self.dire, self.energy)


class SeamCarvingUi(QWidget):
    def __init__(self, parent=None):
        super(SeamCarvingUi, self).__init__(parent)
        self.setWindowTitle("seam carving, 2020/12/04, Tsinghua, Beijing")
        self.pix = QPixmap("data/anonymous.jpg")
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.load_img_btn = QPushButton(self)
        self.save_img_btn = QPushButton(self)

        self.size_group = QButtonGroup(self)
        self.width_select = QRadioButton("width", self)
        self.width_select.setChecked(True)
        self.height_select = QRadioButton("height", self)
        self.size_group.addButton(self.width_select, 11)
        self.size_group.addButton(self.height_select, 12)

        self.energy_group = QButtonGroup(self)
        self.e1_energy_select = QRadioButton("e1 energy", self)
        self.forward_energy_select = QRadioButton("forward energy", self)
        self.laplacian_energy_select = QRadioButton("lap energy", self)
        self.hog_energy_select = QRadioButton("HOG energy", self)
        self.e1_energy_select.setChecked(True)
        self.energy_group.addButton(self.e1_energy_select, 21)
        self.energy_group.addButton(self.forward_energy_select, 22)
        self.energy_group.addButton(self.laplacian_energy_select, 23)
        self.energy_group.addButton(self.hog_energy_select, 24)

        self.resize_btn = QPushButton(self)
        self.pro_bar = QProgressBar(self)
        self.timer = QBasicTimer()
        self.scale_adjust = QDoubleSpinBox(self)
        self.scale_adjust.setSingleStep(0.01)

        self.raw_image = 0

        self.d_width = 1700
        self.d_height = 700

        self.init_view()
        self.seam_carving = seamCarving.SeamCarving()
        self.input_path = ""
        self.output_path = ""
        self.thread = 0

        self.all_time = 0
        self.now_time = 0

        self.select = "c"

        self.start_mark = False

        self.energy_select = "e1"

    def init_view(self):
        self.resize(self.d_width, self.d_height)
        self.pix = QPixmap()

        self.load_img_btn.setText("Load Image")
        self.load_img_btn.resize(100, 30)
        self.load_img_btn.move(int(self.d_width - 190), 100)
        self.load_img_btn.clicked.connect(self.on_load_image_btn)

        self.save_img_btn.setText("Save Image")
        self.save_img_btn.resize(100, 30)
        self.save_img_btn.move(int(self.d_width - 190), 150)
        self.save_img_btn.clicked.connect(self.on_save_img_btn)

        self.width_select.resize(100, 30)
        self.width_select.move(int(self.d_width - 100), 210)

        self.height_select.resize(100, 30)
        self.height_select.move(int(self.d_width - 230), 210)

        self.scale_adjust.resize(100, 30)
        self.scale_adjust.move(int(self.d_width - 190), 260)

        self.e1_energy_select.resize(100, 30)
        self.e1_energy_select.move(int(self.d_width - 230), 300)

        self.forward_energy_select.resize(120, 30)
        self.forward_energy_select.move(int(self.d_width - 130), 300)

        self.laplacian_energy_select.resize(120, 30)
        self.laplacian_energy_select.move(int(self.d_width - 230), 340)

        self.hog_energy_select.resize(120, 30)
        self.hog_energy_select.move(int(self.d_width - 130), 340)

        self.resize_btn.setText("resize")
        self.resize_btn.resize(100, 30)
        self.resize_btn.move(int(self.d_width - 190), 380)
        self.resize_btn.clicked.connect(self.on_resize_btn)

        self.pro_bar.resize(220, 20)
        self.pro_bar.move(int(self.d_width - 235), 420)

        self.size_group.buttonClicked.connect(self.on_radio_choose)
        self.energy_group.buttonClicked.connect(self.on_radio_choose)

    def on_radio_choose(self):
        sender = self.sender()
        if sender == self.size_group:
            if self.size_group.checkedId() == 11:
                self.select = "c"
            elif self.size_group.checkedId() == 12:
                self.select = "r"
        elif sender == self.energy_group:
            if self.energy_group.checkedId() == 21:
                self.energy_select = "e1"
            elif self.energy_group.checkedId() == 22:
                self.energy_select = "forward"
            elif self.energy_group.checkedId() == 23:
                self.energy_select = "laplacian"
            elif self.energy_group.checkedId() == 24:
                self.energy_select = "hog"

    def on_load_image_btn(self):
        absolute_path = QFileDialog.getOpenFileName(self, 'Open file',
                                                    './', "ALL (*.*)")
        if absolute_path:
            cur_path = QDir('.')
            relative_path = cur_path.relativeFilePath(absolute_path[0])
            self.input_path = relative_path
            self.raw_image = cv2.imread(self.input_path)

    def on_resize_btn(self):
        if not self.timer.isActive():
            self.all_time = self.seam_carving.get_all_time(self.raw_image, self.scale_adjust.value())
            self.pro_bar.setMaximum(self.all_time)
            self.timer.start(400, self)
            self.start_mark = True
            self.thread = MyThread(self.seam_carving, self.raw_image, self.scale_adjust.value(), self.select,
                                   self.energy_select)
            self.thread.start()

    def on_save_img_btn(self):
        filename = QFileDialog.getSaveFileName(self, 'save', "*.jpg")
        if filename:
            try:
                img = cv2.imread("data/anonymous.jpg")
                cur_path = QDir('.')
                relative_path = cur_path.relativeFilePath(filename[0])
                cv2.imwrite(relative_path, img)
            except:
                pass

    def paintEvent(self, event):
        if self.start_mark:
            self.pix = QPixmap("data/anonymous.jpg")
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self.pix)

    def timerEvent(self, e):
        self.now_time = self.seam_carving.get_now_time()
        img = self.seam_carving.get_now_image()
        if self.select == "c":
            cv2.imwrite("data/anonymous.jpg", img)
        else:
            cv2.imwrite("data/anonymous.jpg", img)
        self.update()
        self.pro_bar.setValue(self.now_time)

        if self.seam_carving.get_end_mark():
            self.timer.stop()

    def changeEvent(self, e):
        if e.type() == QtCore.QEvent.WindowStateChange:
            self.d_width = self.width()
            self.d_height = self.height()
            self.init_view()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = SeamCarvingUi()
    form.show()
    sys.exit(app.exec_())
