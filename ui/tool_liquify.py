import numpy as np
import cv2


class tool_liquify:
	def __init__(self, img_width, img_height, brushWidth, scale):
		self.img_width = img_width
		self.img_height = img_height
		self.scale = float(scale)
		self.mask_template_x = np.zeros((img_height, img_width), np.float32)
		self.mask_template_y = np.zeros((img_height, img_width), np.float32)
		for i in range(img_height):
			for j in range(img_width):
				self.mask_template_x[i][j] = j
				self.mask_template_y[i][j] = i
		self.mask_x = np.copy(self.mask_template_x)
		self.mask_y = np.copy(self.mask_template_y)
		self.width = brushWidth
		self.strength = 0.6

	def liquid(self, radius, current_x, current_y, last_x, last_y):
		r2 = radius*radius
		width = self.img_width
		height = self.img_height
		new_mask_x = np.copy(self.mask_x)
		new_mask_y = np.copy(self.mask_y)
		for i in range(max(last_y-radius,0),min(last_y+radius+1,height)):
			for j in range(max(last_x-radius,0),min(last_x+radius+1,width)):
				d2 =(j - last_x)*(j - last_x) + (i - last_y)*(i - last_y)
				if (r2 >= d2):
					offset = self.strength*(1-d2/r2)
					sx = j - offset * (current_x - last_x)
					sy = i - offset * (current_y - last_y)
					if (sx<0): sx=0
					if (sx>=width-1): sx = width-2
					if (sy<0): sy=0
					if (sy>=height-1): sy = height-2
					sqx = int(sx)
					srx = sx - sqx
					sqy = int(sy)
					sry = sy - sqy
					new_mask_x[i][j] = (1-srx)*(1-sry)*self.mask_x[sqy][sqx] + srx*(1-sry)*self.mask_x[sqy][sqx+1] + (1-srx)*sry*self.mask_x[sqy+1][sqx] + srx*sry*self.mask_x[sqy+1][sqx+1]
					new_mask_y[i][j] = (1-srx)*(1-sry)*self.mask_y[sqy][sqx] + srx*(1-sry)*self.mask_y[sqy][sqx+1] + (1-srx)*sry*self.mask_y[sqy+1][sqx] + srx*sry*self.mask_y[sqy+1][sqx+1]
		self.mask_x = new_mask_x
		self.mask_y = new_mask_y
	
	def liquify(self, origin):
		#sacrifice anti-alias for speed
		return origin[self.mask_y.astype(np.uint8),self.mask_x.astype(np.uint8)]

	def update(self, image, mask, points):
		num_pnts = len(points)
		radius = self.width/self.scale/2
		self.liquid(int(radius), int(points[num_pnts-1].x()/self.scale), int(points[num_pnts-1].y()/self.scale), int(points[num_pnts-2].x()/self.scale), int(points[num_pnts-2].y()/self.scale))
		return self.liquify(image),self.liquify(mask)

	def update_width(self, d):
		self.width = min(160, max(20, self.width+ d))
		return self.width

	def reset(self):
		self.mask_x = np.copy(self.mask_template_x)
		self.mask_y = np.copy(self.mask_template_y)
