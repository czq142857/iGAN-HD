import numpy as np
import cv2


class tool_blend:
	def __init__(self, img_width, img_height, brushWidth, scale):
		self.img_width = img_width
		self.img_height = img_height
		self.scale = float(scale)
		self.width = brushWidth
	
	def update(self, image, points, color, ratio):
		img = np.copy(image)
		num_pnts = len(points)
		w = int(max(1, self.width / self.scale))
		c = (color.red(), color.green(), color.blue())
		for i in range(0, num_pnts - 1):
			pnt1 = (int(points[i].x() / self.scale), int(points[i].y() / self.scale))
			pnt2 = (int(points[i + 1].x() / self.scale), int(points[i + 1].y() / self.scale))
			cv2.line(img, pnt1, pnt2, c, w)
		if num_pnts==1:
			pnt1 = (int(points[0].x() / self.scale), int(points[0].y() / self.scale))
			pnt2 = (int(points[0].x() / self.scale), int(points[0].y() / self.scale))
			cv2.line(img, pnt1, pnt2, c, w)
		img = (image*(1-ratio)+img*ratio).astype(np.uint8)
		return img
	
	def update_width(self, d):
		self.width = min(100, max(1, self.width+ d))
		return self.width

	def reset(self):
		return
