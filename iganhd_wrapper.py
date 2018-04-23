from __future__ import print_function
from time import time
import numpy as np
import sys
from PyQt5.QtCore import *
import cv2

class iganhd_wrapper(QThread):

	def __init__(self, model, FLAGS):
		QThread.__init__(self)
		self.model = model
		self.img_width = FLAGS.output_width
		self.img_height = FLAGS.output_height
		self.result_img = np.full((self.img_height,self.img_width,3), 255, np.uint8)

	def generate(self, image, mask, edge):
		color_flag = self.check_notblack(mask)
		mask_flag = self.check_minor(mask) and color_flag
		edge_flag = self.check_notwhite(edge)
		if color_flag:
			img = image/127.5 - 1
		else:
			img = None
		if mask_flag:
			msk = mask/255.0
		else:
			msk = None
		if edge_flag:
			ege = edge/127.5 - 1
		else:
			ege = None
		print("\noptimization inputs: image ",color_flag,", mask ",mask_flag,", edge ",edge_flag,"\n")
		if color_flag or edge_flag:
			img = self.model.predict(color_map=img, mask=msk, edge_map = ege)
			self.result_img = ((img[0]+1)*128).astype(np.uint8)
	
	def check_notblack(self, image):
		maxima = np.max(image)
		if maxima<5:
			return False
		return True
	
	def check_notwhite(self, image):
		minima = np.min(image)
		if minima>250:
			return False
		return True
	
	def check_minor(self, image):
		ave = np.mean(image)
		if ave>192:
			return False
		return True
	
	def run(self):
		time_to_wait = 50 #ms
		while (1):
			self.msleep(time_to_wait)

	def reset(self):
		self.result_img = np.full((self.img_height,self.img_width,3), 255, np.uint8)
	
	
	
