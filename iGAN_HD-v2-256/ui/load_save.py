import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog

def save_image(image):
	save_dir = QFileDialog.getSaveFileName(None, 'Select a folder to save the image', '.', 'PNG (*.png);;JPG(*.jpg);;BMP (*.bmp)')
	save_dir = str(save_dir[0])
	if save_dir=="": return
	if image is None: return
	print('save to (%s)' % save_dir)
	cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def save_image_rgba(image, mask):
	save_dir = QFileDialog.getSaveFileName(None, 'Select a folder to save the image', '.', 'PNG (*.png)')
	save_dir = str(save_dir[0])
	if save_dir=="": return
	if (image is None) or (mask is None): return
	print('save to (%s)' % save_dir)
	img_RGBA = cv2.merge((image[:,:,2], image[:,:,1], image[:,:,0], mask[:,:,0]))
	cv2.imwrite(save_dir, img_RGBA)

def load_image():
	load_dir = QFileDialog.getOpenFileName(None, 'Select an image to load', '.')
	load_dir = str(load_dir[0])
	print('load from (%s)' % load_dir)
	image = cv2.imread(load_dir, cv2.IMREAD_COLOR)
	if image is None: return None
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_image_rgba():
	load_dir = QFileDialog.getOpenFileName(None, 'Select an image to load', '.')
	load_dir = str(load_dir[0])
	print('load from (%s)' % load_dir)
	image = cv2.imread(load_dir, cv2.IMREAD_UNCHANGED)
	if image is None: return None,None
	image_shape = image.shape
	if (len(image_shape)==2):
		return cv2.merge((image,image,image)),np.full((image_shape[0],image_shape[1],3), 255, np.uint8)
	elif (image_shape[2]==3):
		return cv2.cvtColor(image, cv2.COLOR_BGR2RGB),np.full((image_shape[0],image_shape[1],3), 255, np.uint8)
	elif (image_shape[2]==4):
		return cv2.cvtColor(image[:,:,0:3], cv2.COLOR_BGR2RGB),cv2.merge((image[:,:,3],image[:,:,3],image[:,:,3]))
	return None,None
	
	
	