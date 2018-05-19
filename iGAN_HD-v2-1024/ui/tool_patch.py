import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
from scipy.sparse.linalg import lsqr
import cv2


class tool_patch:
	def __init__(self, img_width, img_height, scale):
		self.img_width = img_width
		self.img_height = img_height
		self.scale = float(scale)
		self.auxiliary_template = np.full((self.img_height,self.img_width,3), 255, np.uint8)
		self.auxiliary_template[:,:,2] = self.auxiliary_template[:,:,2]-255 #(255,255,0) -> (min, min, max) blue
		self.blendmask = None
		self.blendarea = None
		self.blendareamask = None
		self.points = []
		self.blendphase = 0
		self.width = 1


	def update(self, image, mask, points):
		if self.blendphase==0:
			aux = np.copy(self.auxiliary_template)
			num_pnts = len(points)
			c = (0,0,255)
			for i in range(0, num_pnts-1):
				pnt1 = (int(points[i].x() / self.scale), int(points[i].y() / self.scale))
				pnt2 = (int(points[i+1].x() / self.scale), int(points[i+1].y() / self.scale))
				cv2.line(aux, pnt1, pnt2, c, 1)
			return image,mask,aux
		elif self.blendphase==1:
			self.compute_mask(image, mask, points)
			self.blendphase = 2
			aux = np.copy(self.auxiliary_template)
			num_pnts = len(self.points)
			c = (0,0,255)
			self.fill_aux(aux,0,0)
			for i in range(0, num_pnts-1):
				pnt1 = (self.points[i][0],self.points[i][1])
				pnt2 = (self.points[i+1][0],self.points[i+1][1])
				cv2.line(aux, pnt1, pnt2, c, 1)
			return image,mask,aux
		elif self.blendphase==2:
			if len(points)>=2:
				offset_x = int( (points[len(points)-1].x() - points[0].x()) / self.scale )
				offset_y = int( (points[len(points)-1].y() - points[0].y()) / self.scale )
			else:
				offset_x = 0
				offset_y = 0
			aux = np.copy(self.auxiliary_template)
			num_pnts = len(self.points)
			c = (0,0,255)
			self.fill_aux(aux,offset_x,offset_y)
			for i in range(0, num_pnts-1):
				pnt1 = (self.points[i][0] + offset_x, self.points[i][1] + offset_y)
				pnt2 = (self.points[i+1][0] + offset_x, self.points[i+1][1] + offset_y)
				cv2.line(aux, pnt1, pnt2, c, 1)
			return image,mask,aux
		else:
			if len(points)>=2:
				offset_x = int( (points[len(points)-1].x() - points[0].x()) / self.scale )
				offset_y = int( (points[len(points)-1].y() - points[0].y()) / self.scale )
			else:
				offset_x = 0
				offset_y = 0
			img,msk = self.compute_blend(image, mask, offset_x, offset_y)
			self.blendphase = 2
			aux = np.copy(self.auxiliary_template)
			num_pnts = len(self.points)
			c = (0,0,255)
			self.fill_aux(aux,0,0)
			for i in range(0, num_pnts-1):
				pnt1 = (self.points[i][0],self.points[i][1])
				pnt2 = (self.points[i+1][0],self.points[i+1][1])
				cv2.line(aux, pnt1, pnt2, c, 1)
			return img,msk,aux
	
	def compute_mask(self, image, mask, points):
		num_pnts = len(points)
		self.points = []
		for i in range(0, num_pnts):
			self.points.append([int(points[i].x() / self.scale), int(points[i].y() / self.scale)])
		self.points.append([int(points[0].x() / self.scale), int(points[0].y() / self.scale)]) #close loop
		self.blendmask = np.zeros((self.img_height,self.img_width), np.uint8)
		cv2.fillPoly(self.blendmask, np.array([self.points], np.int32), 255)
		self.blendarea = np.copy(image)
		self.blendareamask = np.copy(mask)
		return
	
	def compute_blend(self, image, mask, offset_x, offset_y):
		print("This could take a while ... depending on how many pixels you selected")
		r = self.poisson_blend(image[:,:,0],mask[:,:,0],self.blendmask,self.blendarea[:,:,0],self.blendareamask[:,:,0],offset_x,offset_y)
		g = self.poisson_blend(image[:,:,1],mask[:,:,1],self.blendmask,self.blendarea[:,:,1],self.blendareamask[:,:,1],offset_x,offset_y)
		b = self.poisson_blend(image[:,:,2],mask[:,:,2],self.blendmask,self.blendarea[:,:,2],self.blendareamask[:,:,2],offset_x,offset_y)
		m = self.compute_blendmask(mask[:,:,0],self.blendmask,self.blendareamask[:,:,0],offset_x,offset_y)
		return cv2.merge((r,g,b)),cv2.merge((m,m,m))
	
	def poisson_blend(self, im_t, mask_t, mask_b, im_s, mask_s, offset_x, offset_y):
		imw = self.img_width
		imh = self.img_height
		#count non-empty pixels
		e_max = 0
		for y in range(imh):
			for x in range(imw):
				if mask_b[y][x]>127 and mask_s[y][x]>127:
				   e_max = e_max+1
		#prepare huge matrices
		row_counter = 0
		data_counter = 0
		data = np.zeros(e_max*4*2, np.int)
		row = np.zeros(e_max*4*2, np.int)
		col = np.zeros(e_max*4*2, np.int)
		b = np.zeros((e_max*4), np.int)
		ref_flag = False
		print('constructing A & b')
		for y in range(max(1,1-offset_y),min(imh-1,imh-1-offset_y)):
			for x in range(max(1,1-offset_x),min(imw-1,imw-1-offset_x)):
				if mask_b[y][x]>127 and mask_s[y][x]>127:
				
					#minimize ((v(x,y)-vt(x-1,y)) - (s(x,y)-s(x-1,y)))^2
					if mask_b[y][x-1]>127:
						row[data_counter] = row_counter
						col[data_counter] = y*imw+x
						data[data_counter] = 1
						data_counter+=1
						row[data_counter] = row_counter
						col[data_counter] = y*imw+x-1
						data[data_counter] = -1
						data_counter+=1
						b[row_counter] = int(im_s[y][x])-im_s[y][x-1]
						row_counter+=1
					elif mask_t[y+offset_y][x-1+offset_x]>127:
						row[data_counter] = row_counter
						col[data_counter] = y*imw+x
						data[data_counter] = 1
						data_counter+=1
						b[row_counter] = int(im_s[y][x])-im_s[y][x-1]+im_t[y+offset_y][x-1+offset_x]
						ref_flag = True
						row_counter+=1
					
					#minimize ((v(x,y)-vt(x+1,y)) - (s(x,y)-s(x+1,y)))^2
					if mask_b[y][x+1]>127:
						row[data_counter] = row_counter
						col[data_counter] = y*imw+x
						data[data_counter] = 1
						data_counter+=1
						row[data_counter] = row_counter
						col[data_counter] = y*imw+x+1
						data[data_counter] = -1
						data_counter+=1
						b[row_counter] = int(im_s[y][x])-im_s[y][x+1]
						row_counter+=1
					elif mask_t[y+offset_y][x+1+offset_x]>127:
						row[data_counter] = row_counter
						col[data_counter] = y*imw+x
						data[data_counter] = 1
						data_counter+=1
						b[row_counter] = int(im_s[y][x])-im_s[y][x+1]+im_t[y+offset_y][x+1+offset_x]
						ref_flag = True
						row_counter+=1
					
					#minimize ((v(x,y)-vt(x,y-1)) - (s(x,y)-s(x,y-1)))^2
					if mask_b[y-1][x]>127:
						row[data_counter] = row_counter
						col[data_counter] = y*imw+x
						data[data_counter] = 1
						data_counter+=1
						row[data_counter] = row_counter
						col[data_counter] = (y-1)*imw+x
						data[data_counter] = -1
						data_counter+=1
						b[row_counter] = int(im_s[y][x])-im_s[y-1][x]
						row_counter+=1
					elif mask_t[y-1+offset_y][x+offset_x]>127:
						row[data_counter] = row_counter
						col[data_counter] = y*imw+x
						data[data_counter] = 1
						data_counter+=1
						b[row_counter] = int(im_s[y][x])-im_s[y-1][x]+im_t[y-1+offset_y][x+offset_x]
						ref_flag = True
						row_counter+=1
					
					#minimize ((v(x,y)-vt(x,y+1)) - (s(x,y)-s(x,y+1)))^2
					if mask_b[y+1][x]>127:
						row[data_counter] = row_counter
						col[data_counter] = y*imw+x
						data[data_counter] = 1
						data_counter+=1
						row[data_counter] = row_counter
						col[data_counter] = (y+1)*imw+x
						data[data_counter] = -1
						data_counter+=1
						b[row_counter] = int(im_s[y][x])-im_s[y+1][x]
						row_counter+=1
					elif mask_t[y+1+offset_y][x+offset_x]>127:
						row[data_counter] = row_counter
						col[data_counter] = y*imw+x
						data[data_counter] = 1
						data_counter+=1
						b[row_counter] = int(im_s[y][x])-im_s[y+1][x]+im_t[y+1+offset_y][x+offset_x]
						ref_flag = True
						row_counter+=1
		
		print('computing ...')
		if ref_flag:
			#compute least square
			data = data[0:data_counter]
			row = row[0:data_counter]
			col = col[0:data_counter]
			b = b[0:row_counter]
			A = csr_matrix((data, (row, col)), shape=(row_counter, imh*imw))
			solution = lsqr(A, b)[0]
			solution = np.clip(np.reshape(solution, (imh,imw)), 0, 255).astype(np.uint8)
		else:
			#direct copy
			solution = im_s
		
		img = np.copy(im_t)
		for y in range(max(1,1-offset_y),min(imh-1,imh-1-offset_y)):
			for x in range(max(1,1-offset_x),min(imw-1,imw-1-offset_x)):
				if mask_b[y][x]>127 and mask_s[y][x]>127:
					img[y+offset_y][x+offset_x] = solution[y][x]
		return img
		
	def compute_blendmask(self, mask_t, mask_b, mask_s, offset_x, offset_y):
		msk = np.copy(mask_t)
		imw = self.img_width
		imh = self.img_height
		for y in range(max(1,1-offset_y),min(imh-1,imh-1-offset_y)):
			for x in range(max(1,1-offset_x),min(imw-1,imw-1-offset_x)):
				if mask_b[y][x]>127 and mask_s[y][x]>127:
					msk[y+offset_y][x+offset_x] = 255
		return msk
	
	def fill_aux(self, aux, offset_x, offset_y):
		imw = self.img_width
		imh = self.img_height
		for y in range(max(1,1-offset_y),min(imh-1,imh-1-offset_y)):
			for x in range(max(1,1-offset_x),min(imw-1,imw-1-offset_x)):
				if self.blendmask[y][x]>127 and self.blendareamask[y][x][0]>127:
					aux[y+offset_y][x+offset_x] = self.blendarea[y][x]
	
	def next(self):
		self.blendphase = self.blendphase+1
		
	def update_width(self, d):
		self.width = 1
		return self.width
		
	def reset(self):
		self.blendmask = None
		self.blendarea = None
		self.blendareamask = None
		self.points = []
		self.blendphase = 0
