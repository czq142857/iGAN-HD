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
		self.slider_step = FLAGS.slider_step
		self.keep_img = np.full((1,self.img_height,self.img_width,3), 255, np.uint8)
		self.result_img = self.keep_img[0]
		
		self.run_flag = False
		self.window = None
		self.image = None
		self.mask = None
		self.edge = None
	
	def get_image_slider(self, num):
		if not self.run_flag:
			num = int(num*(len(self.keep_img)-1)/self.slider_step)
			self.result_img = self.keep_img[num]
	
	def generate(self, window, image, mask, edge):
		self.run_flag = True
		self.window = window
		self.image = image
		self.mask = mask
		self.edge = edge
	
	def run(self):
		time_to_wait = 50 #ms
		while (1):
			if (self.run_flag):
				self.generate_run(self.window, self.image, self.mask, self.edge)
				self.run_flag = False
				self.window = None
				self.image = None
				self.mask = None
				self.edge = None
			self.msleep(time_to_wait)

	def generate_run(self, window, image, mask, edge):
		def update_image_signal(img):
			self.result_img = img
			self.keep_img = np.concatenate((self.keep_img, [img]), axis=0)
			window.update_slider.emit(self.slider_step)
			window.update_image_signal()
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
			self.keep_img = np.zeros((0,self.img_height,self.img_width,3),np.uint8)
			self.model.predict(update_image_signal, color_map=img, mask=msk, edge_map = ege)
	
	def check_notblack(self, image):
		maxima = np.max(image)
		if maxima<5:
			return False
		return True
	
	def check_notwhite(self, image):
		minima = np.min(image)
		if minima>252:
			return False
		return True
	
	def check_minor(self, image):
		ave = np.mean(image)
		if ave>222:
			return False
		return True

	def reset(self):
		self.keep_img = np.full((1,self.img_height,self.img_width,3), 255, np.uint8)
		self.result_img = self.keep_img[0]
		self.run_flag = False
		self.window = None
		self.image = None
		self.mask = None
		self.edge = None

	
	
	
