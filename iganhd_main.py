import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from ui.gui_design import gui_design
from model_def.model import BranchGAN
from model_def.im2z import IM2Z
from iganhd_wrapper import iganhd_wrapper
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def parse_args():
	flags = tf.app.flags
	#BranchGAN params
	flags.DEFINE_integer("epoch", 20, "Epoch to train [25]")
	flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
	flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
	flags.DEFINE_integer("batch_size", 20, "The size of batch images [64]")
	flags.DEFINE_integer("input_height", 256, "The size of image to use (will be center cropped). [108]")
	flags.DEFINE_integer("input_width", 256, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
	flags.DEFINE_integer("output_height", 256, "The size of the output images to produce [64]")
	flags.DEFINE_integer("output_width", 256, "The size of the output images to produce. If None, same value as output_height [None]")
	flags.DEFINE_string("dataset", "celeba_hq256", "The name of dataset [celebA, mnist, lsun]")
	flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
	flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
	flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
	flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
	flags.DEFINE_integer("z_dim", 30, "Dimensions of z [50]")
	flags.DEFINE_boolean("use_z_pyramid", True, "True for using z pyramid")
	flags.DEFINE_boolean("use_residual_block", False, "True for using residual block")
	flags.DEFINE_boolean("use_two_stage_training", True, "True for using two-stage training at each epoch")
	flags.DEFINE_boolean("random_flip", False, "True for randomly flipping training images")
	flags.DEFINE_boolean("random_crop", False, "True for randomly cropping training images")
	flags.DEFINE_boolean("random_rotate", False, "True for randomly rotating training images")
	flags.DEFINE_boolean("train_im2z", False, "True for training im2z net, False for testing im2z net.")
	flags.DEFINE_boolean("im2z", False, "True for opening im2z mode")
	#UI params
	flags.DEFINE_integer("win_width", 512, "the size of the main window [512]")
	return flags.FLAGS

if __name__ == '__main__':
	FLAGS = parse_args()
	
	#important: don't remove this part, or pyqt will punish you
	#make the width a multiple of 4
	FLAGS.win_width = int(FLAGS.win_width/4)*4
	FLAGS.win_height = int(FLAGS.win_width*FLAGS.output_height/FLAGS.output_width)

	#from BranchGAN
	if not FLAGS.input_width:
		exit("[Exit] input_width is None. please use flag '--input_width' to specify the input image width.")
	else:
		if not FLAGS.output_width:
			FLAGS.output_width = FLAGS.input_width
			FLAGS.crop = False
		elif FLAGS.output_width < FLAGS.input_width:
			FLAGS.crop = True
		elif FLAGS.output_width == FLAGS.input_width:
			FLAGS.crop = False
		elif FLAGS.output_width > FLAGS.input_width:
			exit("[Exit] output_width should be smaller than or equal to input_width")

	if FLAGS.input_height is None:
		FLAGS.input_height = FLAGS.input_width

	if FLAGS.output_height is None:
		FLAGS.output_height = FLAGS.input_height
	elif FLAGS.output_height > FLAGS.input_height:
		exit("[Exit] output_height should be smaller than or equal to input_height")
	elif FLAGS.output_height < FLAGS.input_height:
		FLAGS.crop = True
	
	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		branchgan = BranchGAN(
			sess,
			input_width=FLAGS.input_width,
			input_height=FLAGS.input_height,
			output_width=FLAGS.output_width,
			output_height=FLAGS.output_height,
			batch_size=FLAGS.batch_size,
			z_dim=FLAGS.z_dim,
			dataset_name=FLAGS.dataset,
			input_fname_pattern=FLAGS.input_fname_pattern,
			crop=FLAGS.crop,
			checkpoint_dir=FLAGS.checkpoint_dir,
			sample_dir=FLAGS.sample_dir, 
			use_z_pyramid=FLAGS.use_z_pyramid, 
			use_residual_block = FLAGS.use_residual_block,
			use_two_stage_training=FLAGS.use_two_stage_training,
			random_crop=FLAGS.random_crop,
			random_flip=FLAGS.random_flip,
			random_rotate=FLAGS.random_rotate,
			epoch = FLAGS.epoch)
		
		im2z=IM2Z(FLAGS, branchgan)
		im2z.init_states()
		
		iganhd_wrapper = iganhd_wrapper(im2z, FLAGS)
		
		# initialize application
		app = QApplication(sys.argv)
		window = gui_design(iganhd_wrapper, FLAGS)
		app.setWindowIcon(QIcon('logo.png'))
		window.setWindowTitle('iGAN-HD')
		window.setWindowFlags(window.windowFlags() & ~Qt.WindowMaximizeButtonHint)
		window.show()
		app.exec_()
	