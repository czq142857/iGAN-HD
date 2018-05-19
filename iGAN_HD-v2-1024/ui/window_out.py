import numpy as np
import time
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from .load_save import save_image

class window_out(QWidget):

	def __init__(self, wrapper, FLAGS):
		QWidget.__init__(self)
		self.wrapper = wrapper
		self.width = FLAGS.win_width
		self.height = FLAGS.win_height
		self.current_img = self.wrapper.result_img
		self.move(self.width, self.height)

	def reset(self):
		self.current_img = self.wrapper.result_img
		self.update()

	def paintEvent(self, event):
		painter = QPainter()
		painter.begin(self)
		painter.fillRect(event.rect(), Qt.white)

		bigim = cv2.resize(self.current_img, (self.width, self.height))
		qImg = QImage(bigim.tostring(), self.width, self.height, QImage.Format_RGB888)
		painter.drawImage(0, 0, qImg)

		painter.end()

	def save(self):
		save_image(self.current_img);



