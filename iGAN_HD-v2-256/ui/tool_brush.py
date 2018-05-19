import numpy as np
import cv2


class tool_brush:
	def __init__(self, img_width, img_height, brushWidth, scale):
		self.img_width = img_width
		self.img_height = img_height
		self.scale = float(scale)
		self.width = brushWidth


	def update(self, image, mask, points, color):
		img = np.copy(image)
		msk = np.copy(mask)
		num_pnts = len(points)
		w = int(max(1, self.width / self.scale))
		c = (color.red(), color.green(), color.blue())
		white = (255,255,255)
		for i in range(0, num_pnts - 1):
			pnt1 = (int(points[i].x() / self.scale), int(points[i].y() / self.scale))
			pnt2 = (int(points[i + 1].x() / self.scale), int(points[i + 1].y() / self.scale))
			cv2.line(img, pnt1, pnt2, c, w)
			cv2.line(msk, pnt1, pnt2, white, w)
		if num_pnts==1:
			pnt1 = (int(points[0].x() / self.scale), int(points[0].y() / self.scale))
			pnt2 = (int(points[0].x() / self.scale), int(points[0].y() / self.scale))
			cv2.line(img, pnt1, pnt2, c, w)
			cv2.line(msk, pnt1, pnt2, white, w)
		return img,msk
	
	def updatemask(self, mask, points):
		msk = np.copy(mask)
		num_pnts = len(points)
		w = int(max(1, self.width / self.scale))
		black = (0,0,0)
		for i in range(0, num_pnts - 1):
			pnt1 = (int(points[i].x() / self.scale), int(points[i].y() / self.scale))
			pnt2 = (int(points[i + 1].x() / self.scale), int(points[i + 1].y() / self.scale))
			cv2.line(msk, pnt1, pnt2, black, w)
		if num_pnts==1:
			pnt1 = (int(points[0].x() / self.scale), int(points[0].y() / self.scale))
			pnt2 = (int(points[0].x() / self.scale), int(points[0].y() / self.scale))
			cv2.line(msk, pnt1, pnt2, black, w)
		return msk

	def update_width(self, d):
		self.width = min(100, max(1, self.width+ d))
		return self.width

	def reset(self):
		return
