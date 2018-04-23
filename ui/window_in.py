import numpy as np
import time
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from .load_save import *
from .tool_brush import tool_brush
from .tool_color import tool_color
from .tool_eraser import tool_eraser
from .tool_blend import tool_blend
from .tool_blendvalue import tool_blendvalue
from .tool_liquify import tool_liquify
from .tool_patch import tool_patch
from .tool_mesh import tool_mesh
from .tool_free_mesh import tool_free_mesh

class window_in(QWidget):

	update_color = pyqtSignal(str)
	update_image = pyqtSignal()

	def __init__(self, wrapper, FLAGS):
		QWidget.__init__(self)
		self.wrapper = wrapper
		self.width = FLAGS.win_width
		self.height = FLAGS.win_height
		self.img_width = FLAGS.output_width
		self.img_height = FLAGS.output_height
		self.background = make_background(self.img_width,self.img_height)
		self.true_background = np.full((self.img_height,self.img_width,3), 255, np.uint8)
		self.current_img = np.full((self.img_height,self.img_width,3), 255, np.uint8)
		self.previous_img = np.copy(self.current_img)
		self.current_mask = np.zeros((self.img_height,self.img_width,3), np.uint8)
		self.previous_mask = np.copy(self.current_mask)
		self.current_edge = np.full((self.img_height,self.img_width,3), 255, np.uint8)
		self.previous_edge = np.copy(self.current_edge)
		self.auxiliary = None
		self.temp_img = None
		self.temp_mask = None
		self.combined = np.zeros((self.img_height,self.img_width,3), np.uint8)
		self.layer_color = True
		self.layer_edge = True
		self.layer_bg = True
		self.isPressed = False
		self.points = []
		self.init_color()
		self.pos = None
		self.scale = FLAGS.win_width / float(FLAGS.output_width)
		self.brushWidth = int(8 * self.scale)
		self.type = 'brush'
		self.tool_brush = tool_brush(self.img_width, self.img_height, self.brushWidth, self.scale)
		self.tool_color = tool_color(self.img_width, self.img_height, self.brushWidth, self.scale)
		self.tool_eraser = tool_eraser(self.img_width, self.img_height, self.brushWidth, self.scale)
		self.tool_blend = tool_blend(self.img_width, self.img_height, self.brushWidth, self.scale)
		self.tool_blendvalue = tool_blendvalue(self.img_width, self.img_height, self.brushWidth, self.scale)
		self.tool_liquify = tool_liquify(self.img_width, self.img_height, self.brushWidth*2, self.scale)
		self.tool_patch = tool_patch(self.img_width, self.img_height, self.scale)
		self.tool_mesh = tool_mesh(self.img_width, self.img_height, self.scale)
		self.tool_free_mesh = tool_free_mesh(self.img_width, self.img_height, self.scale)
		self.move(self.width, self.height)
		self.setMouseTracking(True)

	def update_ui(self):
		if self.type=='brush':
			self.current_img,self.current_mask = self.tool_brush.update(self.previous_img, self.previous_mask, self.points, self.color)
		if self.type=='color':
			self.current_img = self.tool_color.update(self.previous_img, self.points, self.color)
		if self.type=='eraser':
			self.current_mask = self.tool_eraser.updatemask(self.previous_mask, self.points)
		if self.type=='blend':
			self.current_img = self.tool_blend.update(self.previous_img, self.points, self.color, 0.3)
		if self.type=='lighten':
			self.current_img = self.tool_blendvalue.update(self.previous_img, self.points, 20)
		if self.type=='darken':
			self.current_img = self.tool_blendvalue.update(self.previous_img, self.points, -20)
		if self.type=='liquify':
			self.current_img,self.current_mask = self.tool_liquify.update(self.previous_img, self.previous_mask, self.points)
		if self.type=='brush_edge':
			self.current_edge = self.tool_brush.updatemask(self.previous_edge, self.points)
		if self.type=='eraser_edge':
			self.current_edge = self.tool_eraser.update(self.previous_edge, self.points)
		if self.type=='patch':
			self.current_img,self.current_mask,self.auxiliary = self.tool_patch.update(self.previous_img, self.previous_mask, self.points)
		if self.type=='patch_from':
			if self.tool_patch.blendphase==0:
				self.current_img,self.current_mask,self.auxiliary = self.tool_patch.update(self.temp_img, self.temp_mask, self.points)
			elif self.tool_patch.blendphase==1:
				self.current_img,self.current_mask,self.auxiliary = self.tool_patch.update(self.temp_img, self.temp_mask, self.points)
				self.temp_img = None
				self.temp_mask = None
				self.current_img,self.current_mask,self.auxiliary = self.tool_patch.update(self.previous_img, self.previous_mask, [])
			else:
				self.current_img,self.current_mask,self.auxiliary = self.tool_patch.update(self.previous_img, self.previous_mask, self.points)
		if self.type=='mesh':
			self.current_img,self.current_mask,self.auxiliary = self.tool_mesh.update(self.previous_img, self.previous_mask, self.points)
		if self.type=='free_mesh':
			self.current_img,self.current_mask,self.auxiliary = self.tool_free_mesh.update(self.previous_img, self.previous_mask, self.points)

	def round_point(self, pnt):
		# print(type(pnt))
		x = int(np.round(pnt.x()))
		y = int(np.round(pnt.y()))
		return QPoint(x, y)

	def init_color(self):
		self.color = QColor(255, 0, 0)  # default color red

	def change_color(self):
		color = QColorDialog.getColor(parent=self)
		self.color = color
		self.update_color.emit(('background-color: %s' % self.color.name()))

	def combine(self):
		if self.layer_color:
			self.combined = np.multiply(self.current_img,self.current_mask/255)
			if self.layer_bg:
				self.combined = self.combined+np.multiply(self.background,1-self.current_mask/255)
			else:
				self.combined = self.combined+np.multiply(self.true_background,1-self.current_mask/255)
			if self.layer_edge:
				self.combined = np.multiply(self.combined,self.current_edge/255)
		else:
			if self.layer_bg:
				self.combined = self.background
			else:
				self.combined = self.true_background
			if self.layer_edge:
				self.combined = np.multiply(self.combined,self.current_edge/255)
		if self.auxiliary is not None:
			self.combined = cv2.merge((np.minimum(self.combined[:,:,0],self.auxiliary[:,:,0]), np.minimum(self.combined[:,:,1],self.auxiliary[:,:,1]), np.maximum(self.combined[:,:,2],self.auxiliary[:,:,2])))
		self.combined = self.combined.astype(np.uint8)
	
	def paintEvent(self, event):
		painter = QPainter()
		painter.begin(self)
		painter.fillRect(event.rect(), Qt.white)

		self.combine()
		bigim = cv2.resize(self.combined, (self.width, self.height))
		qImg = QImage(bigim.tostring(), self.width, self.height, QImage.Format_RGB888)
		painter.drawImage(0, 0, qImg)

		# draw cursor
		if self.pos is not None:
			w = self.brushWidth/2
			c = self.color
			pnt = QPointF(self.pos.x(), self.pos.y())
			if self.type=='brush' or self.type=='brush_edge' or self.type=='color' or self.type=='blend':
				ca = QColor(c.red(), c.green(), c.blue(), 127)
			else:
				ca = QColor(0, 0, 0, 255)

			painter.setPen(QPen(ca, 1))
			if self.type=='brush' or self.type=='brush_edge' or self.type=='color' or self.type=='blend':
				painter.setBrush(ca)
			painter.drawEllipse(pnt, w, w)

		painter.end()

	def wheelEvent(self, event):
		d = event.angleDelta().y() / 100
		if self.type=='brush' or self.type=='brush_edge':
			self.brushWidth = self.tool_brush.update_width(d)
		if self.type=='color':
			self.brushWidth = self.tool_color.update_width(d)
		if self.type=='eraser' or self.type=='eraser_edge':
			self.brushWidth = self.tool_eraser.update_width(d)
		if self.type=='blend':
			self.brushWidth = self.tool_blend.update_width(d)
		if self.type=='lighten' or self.type=='darken':
			self.brushWidth = self.tool_blendvalue.update_width(d)
		if self.type=='liquify':
			self.brushWidth = self.tool_liquify.update_width(d)
		self.update()

	def mousePressEvent(self, event):
		self.pos = self.round_point(event.pos())
		if event.button() == Qt.LeftButton:
			self.isPressed = True
			#backup undo
			if not(self.type=='patch_from' and self.tool_patch.blendphase==0) and self.type!='mesh' and self.type!='free_mesh':
				self.previous_img = self.current_img
				self.previous_mask = self.current_mask
				self.previous_edge = self.current_edge
			self.points.append(self.pos)
			self.update_ui()
		self.update()

	def mouseMoveEvent(self, event):
		self.pos = self.round_point(event.pos())
		if self.isPressed:
			self.points.append(self.pos)
			self.update_ui()
		self.update()

	def mouseReleaseEvent(self, event):
		self.pos = self.round_point(event.pos())
		if event.button() == Qt.LeftButton and self.isPressed:
			if self.type=='brush' or self.type=='brush_edge':
				self.tool_brush.reset()
			if self.type=='color':
				self.tool_color.reset()
			if self.type=='eraser' or self.type=='eraser_edge':
				self.tool_eraser.reset()
			if self.type=='blend':
				self.tool_blend.reset()
			if self.type=='lighten' or self.type=='darken':
				self.tool_blendvalue.reset()
			if self.type=='liquify':
				self.tool_liquify.reset()
			#do not reset patch or mesh if still using
			if self.type=='patch' or self.type=='patch_from':
				self.tool_patch.next()
				self.update_ui()
			if self.type=='mesh':
				self.tool_mesh.next()
				self.update_ui()
			if self.type=='free_mesh':
				self.tool_free_mesh.next()
				self.update_ui()
			self.points = []
			self.isPressed = False
		self.update()

	def generate_result(self):
		self.wrapper.generate(self.current_img,self.current_mask,self.current_edge)
		self.update_image.emit()
		self.update()

	def adopt_result(self):
		self.current_img = np.copy(self.wrapper.result_img)
		self.previous_img = np.copy(self.current_img)
		self.current_mask = np.full((self.img_height,self.img_width,3), 255, np.uint8)
		self.previous_mask = np.copy(self.current_mask)
		self.current_edge = np.full((self.img_height,self.img_width,3), 255, np.uint8)
		self.previous_edge = np.copy(self.current_edge)
		#reset patch and mesh
		self.tool_patch.reset()
		self.tool_mesh.reset()
		self.tool_free_mesh.reset()
		self.update()
	
	def load_color(self):
		self.previous_img = self.current_img
		self.previous_mask = self.current_mask
		image,mask = load_image_rgba()
		if image is None:
			print('Load failed')
			return
		self.current_img = cv2.resize(image, (self.img_width, self.img_height))
		self.current_mask = cv2.resize(mask, (self.img_width, self.img_height))
		if self.type=='patch' or self.type=='patch_from':
			self.use_patch()
		if self.type=='mesh':
			self.use_mesh()
		if self.type=='free_mesh':
			self.use_free_mesh()

	def save_color(self):
		save_image_rgba(self.current_img,self.current_mask);
	
	def undo_color(self):
		self.current_img = np.copy(self.previous_img)
		self.current_mask = np.copy(self.previous_mask)
		self.update()
		if self.type=='mesh':
			self.use_mesh()
		if self.type=='free_mesh':
			self.use_free_mesh()
	
	def clear_color(self):
		self.current_img = np.full((self.img_height,self.img_width,3), 255, np.uint8)
		self.previous_img = np.copy(self.current_img)
		self.current_mask = np.zeros((self.img_height,self.img_width,3), np.uint8)
		self.previous_mask = np.copy(self.current_mask)
		self.update()
	
	def load_edge(self):
		self.previous_edge = self.current_edge
		image = load_image()
		if image is None:
			print('Load failed')
			return
		self.current_edge = cv2.resize(image, (self.img_width, self.img_height))

	def save_edge(self):
		save_image(self.current_edge);
	
	def undo_edge(self):
		self.current_edge = np.copy(self.previous_edge)
		self.update()
	
	def clear_edge(self):
		self.current_edge = np.full((self.img_height,self.img_width,3), 255, np.uint8)
		self.previous_edge = np.copy(self.current_edge)
		self.update()

	def use_brush(self):
		self.type = 'brush'
		self.update_color.emit(('background-color: %s' % self.color.name()))
		self.brushWidth = self.tool_brush.update_width(0)
		self.reset_aux_layer()
		self.update()

	def use_color(self):
		self.type = 'color'
		self.update_color.emit(('background-color: %s' % self.color.name()))
		self.brushWidth = self.tool_color.update_width(0)
		self.reset_aux_layer()
		self.update()

	def use_eraser(self):
		self.type = 'eraser'
		self.brushWidth = self.tool_eraser.update_width(0)
		self.reset_aux_layer()
		self.update()

	def use_blend(self):
		self.type = 'blend'
		self.brushWidth = self.tool_blend.update_width(0)
		self.reset_aux_layer()
		self.update()

	def use_lighten(self):
		self.type = 'lighten'
		self.brushWidth = self.tool_blendvalue.update_width(0)
		self.reset_aux_layer()
		self.update()

	def use_darken(self):
		self.type = 'darken'
		self.brushWidth = self.tool_blendvalue.update_width(0)
		self.reset_aux_layer()
		self.update()

	def use_liquify(self):
		self.type = 'liquify'
		self.brushWidth = self.tool_liquify.update_width(0)
		self.reset_aux_layer()
		self.update()
	
	def use_brush_edge(self):
		self.type = 'brush_edge'
		self.brushWidth = self.tool_brush.update_width(0)
		self.reset_aux_layer()
		self.update()
	
	def use_eraser_edge(self):
		self.type = 'eraser_edge'
		self.brushWidth = self.tool_eraser.update_width(0)
		self.reset_aux_layer()
		self.update()
	
	def use_patch(self):
		self.type = 'patch'
		self.brushWidth = self.tool_patch.update_width(0)
		self.reset_aux_layer()
		self.update()
	
	def use_patch_from(self):
		self.type = 'patch_from'
		self.brushWidth = self.tool_patch.update_width(0)
		self.reset_aux_layer()
		self.previous_img = self.current_img
		self.previous_mask = self.current_mask
		image,mask = load_image_rgba()
		if image is None:
			print('Load failed')
			self.temp_img = self.current_img
			self.temp_mask = self.current_mask
		else:
			self.temp_img = cv2.resize(image, (self.img_width, self.img_height))
			self.temp_mask = cv2.resize(mask, (self.img_width, self.img_height))
		self.update_ui()
		self.update()
	
	def use_mesh(self):
		self.type = 'mesh'
		self.brushWidth = self.tool_mesh.update_width(0)
		self.reset_aux_layer()
		self.previous_img = self.current_img
		self.previous_mask = self.current_mask
		self.update_ui()
		self.update()
	
	def use_free_mesh(self):
		self.type = 'free_mesh'
		self.brushWidth = self.tool_free_mesh.update_width(0)
		self.reset_aux_layer()
		self.previous_img = self.current_img
		self.previous_mask = self.current_mask
		self.update_ui()
		self.update()
	
	def reset_aux_layer(self):
		self.tool_patch.reset()
		self.tool_mesh.reset()
		self.tool_free_mesh.reset()
		if self.temp_img is not None:
			self.temp_img = None
			self.temp_mask = None
			self.current_img = self.previous_img
			self.current_mask = self.previous_mask
		self.auxiliary = None

def make_background(width, height):
	bg = np.full((height,width,3), 255, np.uint8)
	for i in range(height):
		for j in range(width):
			if (int(i/8)+int(j/8))%2==0:
				bg[i][j] = [204,204,204]
	return bg
