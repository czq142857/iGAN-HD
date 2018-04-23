import numpy as np
from scipy.spatial import Delaunay
import cv2


class tool_free_mesh:
	def __init__(self, img_width, img_height, scale):
		self.img_width = img_width
		self.img_height = img_height
		self.scale = float(scale)
		self.auxiliary_template = np.full((self.img_height,self.img_width,3), 255, np.uint8)
		self.auxiliary_template[:,:,2] = self.auxiliary_template[:,:,2]-255 #(255,255,0) -> (min, min, max) blue
		self.mask_x = None
		self.mask_y = None
		self.points_origin = make_points(self.img_width,self.img_height)
		self.points = np.copy(self.points_origin)
		self.point_selected = None
		self.temp_img = None
		self.temp_mask = None
		self.blendphase = 0
		self.width = 1


	def update(self, image, mask, points):
		if self.blendphase==0:
			if len(points)==1:
				self.point_selected = self.get_point_selected(int(points[0].x() / self.scale), int(points[0].y() / self.scale))
			if self.temp_img is None:
				self.temp_img = np.copy(image)
				self.temp_mask = np.copy(mask)
			if self.point_selected is not None:
				self.points[self.point_selected][0] = int(points[len(points)-1].x() / self.scale)
				self.points[self.point_selected][1] = int(points[len(points)-1].y() / self.scale)
			elif len(points)==1:
				newpoints = [int(points[len(points)-1].x()/self.scale),int(points[len(points)-1].y()/self.scale)]
				self.points_origin = np.append(self.points_origin, [self.warp_single_point(newpoints,self.points_origin,self.points,make_tri(self.points_origin))],axis=0)
				self.points = np.append(self.points, [newpoints],axis=0)
				self.point_selected = len(self.points)-1
		else:
			self.blendphase=0
			if self.point_selected is not None:
				self.points[self.point_selected][0] = int(points[len(points)-1].x() / self.scale)
				self.points[self.point_selected][1] = int(points[len(points)-1].y() / self.scale)
			self.point_selected = None
		tri = make_tri(self.points_origin)
		tri.points[:,:] = self.points[:,:]
		
		#liquify
		self.warp(self.points_origin, self.points, tri)
		self.temp_img = self.liquify(image)
		self.temp_mask = self.liquify(mask)
		
		aux = np.copy(self.auxiliary_template)
		tri_index = tri.simplices
		num_tris = len(tri_index)
		c = (0,0,255)
		for i in range(0, num_tris):
			pnt1 = (self.points[tri_index[i][0]][0],self.points[tri_index[i][0]][1])
			pnt2 = (self.points[tri_index[i][1]][0],self.points[tri_index[i][1]][1])
			cv2.line(aux, pnt1, pnt2, c, 1)
			pnt1 = (self.points[tri_index[i][1]][0],self.points[tri_index[i][1]][1])
			pnt2 = (self.points[tri_index[i][2]][0],self.points[tri_index[i][2]][1])
			cv2.line(aux, pnt1, pnt2, c, 1)
			pnt1 = (self.points[tri_index[i][2]][0],self.points[tri_index[i][2]][1])
			pnt2 = (self.points[tri_index[i][0]][0],self.points[tri_index[i][0]][1])
			cv2.line(aux, pnt1, pnt2, c, 1)
		return self.temp_img,self.temp_mask,aux
	
	def get_point_selected(self, x, y):
		for i in range(len(self.points)):
			if abs(x-self.points[i][0])<8 and abs(y-self.points[i][1])<8:
				return i
		return None
	
	def warp(self, im1_pts, im2_pts, tri):
		tri_index = tri.simplices
		tri_num = len(tri_index)
		ps_1 = np.zeros((self.img_height*self.img_width,3), np.float32)
		ps_2 = np.zeros((self.img_height*self.img_width,3), np.float32)
		for i in range(self.img_height):
			for j in range(self.img_width):
				ps_1[i*self.img_width+j][0] = j
				ps_1[i*self.img_width+j][1] = i
				ps_1[i*self.img_width+j][2] = 1
		row_ind = Delaunay.find_simplex(tri,ps_1[:,0:2])
		#transform matrix
		for i in range(tri_num):
			tf_1  = np.array([[im1_pts[tri_index[i][0]][0],im1_pts[tri_index[i][1]][0],im1_pts[tri_index[i][2]][0]],[im1_pts[tri_index[i][0]][1],im1_pts[tri_index[i][1]][1],im1_pts[tri_index[i][2]][1]],[1,1,1]])
			tf_2  = np.array([[im2_pts[tri_index[i][0]][0],im2_pts[tri_index[i][1]][0],im2_pts[tri_index[i][2]][0]],[im2_pts[tri_index[i][0]][1],im2_pts[tri_index[i][1]][1],im2_pts[tri_index[i][2]][1]],[1,1,1]])
			tf_warp = np.dot(tf_1,np.linalg.pinv(tf_2))
			ps_2[row_ind==i,:] = np.dot(ps_1[row_ind==i,:],np.transpose(tf_warp))
		self.mask_x = np.reshape(ps_2[:,0],(self.img_height,self.img_width))
		self.mask_y = np.reshape(ps_2[:,1],(self.img_height,self.img_width))
	
	def warp_single_point(self, point, im1_pts, im2_pts, tri):
		tri_index = tri.simplices
		tri_num = len(tri_index)
		ps_1 = np.zeros((1,3), np.float32)
		ps_2 = np.zeros((1,3), np.float32)
		ps_1[0][0] = point[0]
		ps_1[0][1] = point[1]
		ps_1[0][2] = 1
		i = Delaunay.find_simplex(tri,ps_1[:,0:2])[0]
		#transform matrix
		tf_1  = np.array([[im1_pts[tri_index[i][0]][0],im1_pts[tri_index[i][1]][0],im1_pts[tri_index[i][2]][0]],[im1_pts[tri_index[i][0]][1],im1_pts[tri_index[i][1]][1],im1_pts[tri_index[i][2]][1]],[1,1,1]])
		tf_2  = np.array([[im2_pts[tri_index[i][0]][0],im2_pts[tri_index[i][1]][0],im2_pts[tri_index[i][2]][0]],[im2_pts[tri_index[i][0]][1],im2_pts[tri_index[i][1]][1],im2_pts[tri_index[i][2]][1]],[1,1,1]])
		tf_warp = np.dot(tf_1,np.linalg.pinv(tf_2))
		ps_2[0,:] = np.dot(ps_1[0,:],np.transpose(tf_warp))
		return [int(ps_2[0][0]),int(ps_2[0][1])]
	
	def liquify(self, origin):
		#sacrifice anti-alias for speed
		return origin[self.mask_y.astype(np.uint8),self.mask_x.astype(np.uint8)]
	
	def next(self):
		self.blendphase = self.blendphase+1
		
	def update_width(self, d):
		self.width = 1
		return self.width
		
	def reset(self):
		self.mask_x = None
		self.mask_y = None
		self.points_origin = make_points(self.img_width,self.img_height)
		self.points = np.copy(self.points_origin)
		self.point_selected = None
		self.temp_img = None
		self.temp_mask = None
		self.blendphase = 0
	
def make_points(width, height):
	points = np.zeros((2*2,2), np.int32)
	for i in range(2):
		for j in range(2):
			points[i*2+j][0] = int(j*width)
			points[i*2+j][1] = int(i*height)
	return points

def make_tri(points):
	return Delaunay(points)

