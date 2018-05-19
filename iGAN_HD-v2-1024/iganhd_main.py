import sys
import pickle
import os
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from ui.gui_design import gui_design
#from model_def.model import BranchGAN
from im2z import IM2Z
from iganhd_wrapper import iganhd_wrapper
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf



if __name__ == '__main__':

	class FLAGS:
		def __init__(self):
			self.win_width = 512
			self.win_height = 512
			self.output_width = 1024
			self.output_height = 1024
			self.slider_step = 64

	flags = FLAGS()
	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		#f.InteractiveSession()

		# Import official CelebA-HQ networks.
		with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
			G, D, Gs = pickle.load(file)

		# Generate latent vectors.
		latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
		latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

		# Generate dummy labels (not used by the official networks).
		labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
		images, gen_out, kwds, kwds_in = Gs.run(latents, labels)


		im2z=IM2Z(gen_out, kwds, kwds_in)
		im2z.init_states()

		
		iganhd_wrapper = iganhd_wrapper(im2z, flags.output_height, flags.output_width, flags.slider_step)  #height, width, ...
		
		# initialize application
		app = QApplication(sys.argv)
		window = gui_design(iganhd_wrapper, flags)
		app.setWindowIcon(QIcon('logo.png'))
		window.setWindowTitle('iGAN-HD')
		window.setWindowFlags(window.windowFlags() & ~Qt.WindowMaximizeButtonHint)
		window.show()
		app.exec_()
	