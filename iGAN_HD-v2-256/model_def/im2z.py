from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from tensorflow.python.ops import data_flow_ops


from .ops import *
from .utils import *
from .tf_hog import *


class IM2Z(object):
  def __init__(self, flags, branchgan):

    self.sess = branchgan.sess
    self.flags = flags
    self.branchgan = branchgan

    self.batch_size = flags.batch_size

    self.output_height = flags.output_height
    self.output_width = flags.output_width

    self.z_dim = flags.z_dim

    self.df_dim = branchgan.df_dim

    self.use_z_pyramid = flags.use_z_pyramid
    self.use_residual_block = flags.use_residual_block

    self.checkpoint_dir = flags.checkpoint_dir

    self.c_dim = branchgan.c_dim

    self.build_model()

  def init_states(self):
    tf.global_variables_initializer().run()
    could_load, counter_load = self.load(self.checkpoint_dir)
    if could_load:
      counter = counter_load
      print(" [*] im2z Load SUCCESS")
    else:
      counter = 1
      print(" [!] im2z Load failed...")

    self.build_z_opt_net()
  
  def build_z_opt_net(self):

    self.z_best = tf.Variable(np.zeros([1,self.z_dim * (self.branchgan.n_levels-1)]), dtype=tf.float32, name='best_z')
    if self.use_z_pyramid:
      z_split = tf.split(self.z_best, [self.z_dim] * (self.branchgan.n_levels-1), axis = 1)
      self.G_best = self.branchgan.generator_reuse(z_split)[-1]
    else:
      self.G_best = self.branchgan.generator_reuse(z)[-1]

    self.color_map_placeholder = tf.placeholder(shape=[1,self.output_height,self.output_width,self.c_dim], dtype=tf.float32)
    self.mask_placeholder = tf.placeholder(shape=[1,self.output_height,self.output_width,self.c_dim], dtype=tf.float32)
    self.edge_map_placeholder = tf.placeholder(shape=[1,self.output_height,self.output_width,self.c_dim], dtype=tf.float32)
    self.c_ratio_placeholder = tf.placeholder(dtype=tf.float32)
    self.e_ratio_placeholder = tf.placeholder(dtype=tf.float32)

    self.loss_c = tf.reduce_sum(tf.abs(self.color_map_placeholder - self.G_best)* \
      self.mask_placeholder)/ (tf.reduce_sum( self.mask_placeholder ) + 10.)
    self.loss_e = self.edge_loss(self.G_best,  self.edge_map_placeholder)

    self.loss_best = self.loss_e * self.e_ratio_placeholder + self.loss_c * self.c_ratio_placeholder

    self.best_Z_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss_best, var_list=[self.z_best])
    
    self.clip_z_op = self.z_best.assign(tf.clip_by_value(self.z_best, -1.0, 1.0))

  
  def build_model(self):

    self.image_batch_placeholder = tf.placeholder(tf.float32, \
      shape=(self.batch_size, self.output_height, self.output_width, self.c_dim), name='input_image')

    self.img_placeholder = tf.placeholder(tf.float32, \
      shape=(1, self.output_height, self.output_width, self.c_dim), name='input_img_single')

    self.z_predict = self.z_net(self.image_batch_placeholder)
    self.z_predict_single = self.z_net(self.img_placeholder, reuse=True)
    z_pyramid = tf.split(self.z_predict, [self.z_dim] * (self.branchgan.n_levels-1), axis=1)
    self.imgs_recover = self.branchgan.generator_reuse(z_pyramid)[-1]
    print(self.imgs_recover)
    self.vars_z_net = [var for var in tf.trainable_variables() if 'z_net' in var.name]

    self.loss = tf.reduce_mean(tf.abs(self.imgs_recover - self.image_batch_placeholder))

    self.train_op = tf.train.AdamOptimizer(1e-4, beta1=0.99).minimize(self.loss, var_list=self.vars_z_net) #self.flags.beta1

    self.saver = tf.train.Saver()


  def edge_loss(self, G, edge_map):
    G = tf.image.resize_images(G, [32,32])
    edge_map = tf.image.resize_images(edge_map, [32,32])
    
    edge_map_hog = tf_hog_descriptor(edge_map)
    g_hog = tf_hog_descriptor(G)
    return tf.reduce_mean(tf.abs(g_hog - edge_map_hog))

  def find_best(self, signal_function, z0, image=None, mask=None, edge_map=None):
    time_start = time.time()
    print('[Start] optimizing z')
    
    fdict = {}
    if image is not None:
      ratio_c = 1.0
      fdict[self.color_map_placeholder] = np.expand_dims(image, axis=0)
    else:
      ratio_c = 0.0
      fdict[self.color_map_placeholder] = np.zeros([1, self.output_height, self.output_width, self.c_dim])

    if mask is None:
      mask = np.ones([self.output_height, self.output_width, self.c_dim])
    fdict[self.mask_placeholder] = np.expand_dims(mask, axis=0)

    if edge_map is not None:
      ratio_e = 10.0  #========================================================
      fdict[self.edge_map_placeholder] = np.expand_dims(edge_map, axis = 0)
    else:
      ratio_e = 0.0
      fdict[self.edge_map_placeholder] = np.zeros([1, self.output_height, self.output_width, self.c_dim])

    fdict[self.c_ratio_placeholder]  = ratio_c
    fdict[self.e_ratio_placeholder]  = ratio_e
    
    #self.sess.run(self.z_best.initialized)
    self.sess.run(self.z_best.assign(z0))
    value_o, value = 0, 4
    while np.abs(value - value_o) > 1e-4:
      value_o = value
      value, __ = self.sess.run([self.loss_best, self.best_Z_optimizer], feed_dict=fdict)
      self.sess.run(self.clip_z_op)
      print(value)
	  #czq_flag: if you want to skip some images, say sample 1 out of 4, add code here
      img = self.sess.run(self.G_best, feed_dict=fdict)
      img = ((img[0]+1)*127.999).astype(np.uint8)
      signal_function(img)
    print('[End] optimizing z, time spent ' + '{}'.format(time.time()-time_start))

  def predict(self, signal_function, color_map = None, mask = None, edge_map=None):
    err_c, err_e= 0.0, 0.0
    if color_map is not None and mask is None:
      image = np.expand_dims(color_map, axis=0)
      z0 = self.sess.run(self.z_predict_single, \
        feed_dict={self.img_placeholder:image})
    else:
      if self.use_z_pyramid:
        z0 = np.random.uniform(-1, 1, [1, self.z_dim * (self.branchgan.n_levels-1) ]).astype(np.float32)
      else:
        z0 = np.random.uniform(-1, 1, [1, self.z_dim]).astype(np.float32)
    self.find_best(signal_function, z0, image=color_map, mask=mask, edge_map=edge_map)

  def train(self):
    start_time = time.time()
    
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    could_load, counter_load = self.load(self.checkpoint_dir)
    if could_load:
      counter = counter_load
      print(" [*] im2z Load SUCCESS")
    else:
      counter = 1
      print(" [!] im2z Load failed...")

    could_load, __ = self.branchgan.load(self.branchgan.checkpoint_dir)
    if could_load:
      print(" [*] branchgan Load SUCCESS")
    else:
      print(" [!] branchgan Load failed...")
      exit(1)

    for epoch in xrange(self.flags.epoch):
      batch_idxs = 5000

      self.data = glob(os.path.join(
          "./data", self.branchgan.dataset_name, self.branchgan.input_fname_pattern))
      np.random.shuffle(self.data)
      self.data = np.array(self.data)
      batch_idxs = len(self.data) // self.batch_size

      for idx in xrange(0, batch_idxs):
          batch_imgs = np.asarray([get_image(self.data[i], self.output_height, self.output_width, crop=False) 
            for i in range(idx * self.batch_size ,(idx+1) * self.batch_size, 1)])
          _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.image_batch_placeholder:batch_imgs})

          counter += 1
          print("epoch=%2d batch_nd=%4d/%4d time=%4.4f, loss=%.8f" \
            % (epoch, idx, batch_idxs, time.time() - start_time, loss))

          if np.mod(counter, 500) == 2:
            self.save(self.checkpoint_dir, counter)

  
  def z_net(self, inputs, reuse = False):
      if self.use_z_pyramid:
        z_concat = self.sub_discriminator(inputs, self.z_dim * (self.branchgan.n_levels-1), reuse=reuse)
        z = z_concat 
      else:
          z = self.sub_discriminator(inputs, self.z_dim, reuse=reuse)
      return z

  def residual_block(self, inputs, level, name='d_'):
    if self.use_residual_block:
      df_dim = inputs.get_shape()[3]
      bn1 = instance_batch_norm(name=name+'bn_'+str(level)+'_1')
      h1 = lrelu(bn1(inputs)) #tf.nn.
      h2 = conv2d(h1, df_dim, d_h=1, d_w=1, name=name + 'h_'+str(level)+"_1")

      bn2 = instance_batch_norm(name=name+'bn_'+str(level)+'_2')
      h3 = conv2d(lrelu(bn2(h2)), df_dim, d_h=1, d_w=1, name=name + 'h_'+str(level)+"_2")
      return h3+inputs
    else:
      return inputs

  def sub_discriminator(self, inputs, dims, reuse = False):
    h = inputs
    with tf.variable_scope("z_net", reuse=reuse) as scope:
      for level in range(self.branchgan.n_levels - 1, 0, -1):
          height, width = self.branchgan.height_pyramid[level], self.branchgan.width_pyramid[level]
          df_dim = self.get_f_dim(self.df_dim, height, width)
          print(h, df_dim)
          h = conv2d(h, df_dim, name='znet_h_'+str(level))
          #h = self.residual_block(h, level, name='znet_')
          bn = instance_batch_norm(name='znet_bn_'+str(level))
          h = lrelu(bn(h))
      z = linear(tf.reshape(h, [int(inputs.get_shape()[0]), -1]), dims, 'znet_lin_'+str(self.branchgan.n_levels - 1))              
      return z


  def get_z_dict(self, z_recover):
    if self.use_z_pyramid:
        print(z_recover.shape)
        z_split = np.split(z_recover,  (self.branchgan.n_levels-1), axis=1)
        print(z_split[0].shape)
        z_dict = {self.branchgan.z_pyramid[0]:z_split[0]}
        for level_tmp in range(1, self.branchgan.n_levels-1, 1):
          print(z_split[level_tmp].shape)
          z_dict [self.branchgan.z_pyramid[level_tmp]] = z_split[level_tmp]
    else:
        z_dict = {self.branchgan.z: z_recover}
    return z_dict


  def test(self):

    if self.use_z_pyramid:
        batch_z = []
        batch_z.append(np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32))
        z_dict = {self.branchgan.z_pyramid[0]:batch_z[0]}
        for level_tmp in range(1, self.branchgan.n_levels-1, 1):
          batch_z.append(np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32))
          z_dict [self.branchgan.z_pyramid[level_tmp]] = batch_z[level_tmp]
    else:
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        z_dict = {self.branchgan.z: batch_z}

    imgs = self.sess.run(self.branchgan.G_pyramid[-1], feed_dict=z_dict)
    loss, imgs_recover = self.sess.run([self.loss, self.imgs_recover], feed_dict={self.image_batch_placeholder:imgs})
    print(imgs_recover.shape)
    save_images(imgs, image_manifold_size(self.branchgan.batch_size), './tests/test_img.png')
    save_images(imgs_recover, image_manifold_size(self.branchgan.batch_size), './tests/test_img_recover.png')

    data = glob(os.path.join("./data", self.branchgan.dataset_name, self.branchgan.input_fname_pattern))
    imgs_raw = np.array([get_image(path, self.output_height, self.output_width, crop=False) 
        for path in data[0:self.batch_size]])
    print(imgs_raw.shape)
    loss, imgs_recover = self.sess.run([self.loss, self.imgs_recover], 
        feed_dict={self.image_batch_placeholder:imgs_raw}) 
    print(imgs_recover.shape)
    save_images(imgs_raw, image_manifold_size(self.branchgan.batch_size), './tests/test_raw_imgs.png')
    save_images(imgs_recover, image_manifold_size(self.branchgan.batch_size), './tests/test_raw_imgs_recover.png')
    
    for i in range(self.batch_size):
      img_recover_n, __, __ = self.predict(color_map = imgs_raw[i]) 
      imgs_recover[i] = img_recover_n[0]
    save_images(imgs_raw, image_manifold_size(self.branchgan.batch_size), './tests/test_raw_imgs_bp.png')
    save_images(imgs_recover, image_manifold_size(self.branchgan.batch_size), './tests/test_raw_imgs_recover_bp.png')
    

  
  def get_f_dim(self, f_dim_base, height, width):
      times = int(2 **(8-math.log((height + width)//2, 2)))
      if times > 8:
          times = 8
      elif times < 1:
          times = 1
      return times * f_dim_base
  
  @property
  def model_dir(self):
    return "im2z_{}-{}-{}-{}_{}-{}-{}_{}-{}-{}_{}-{}".format(
        self.branchgan.dataset_name, self.branchgan.n_levels, 
        self.branchgan.output_height, self.branchgan.output_width,
        self.branchgan.use_z_pyramid, self.branchgan.z_dim,
        self.branchgan.df_dim, self.branchgan.gf_dim, self.branchgan.use_residual_block,
        self.branchgan.batch_size, self.branchgan.epoch, self.branchgan.use_two_stage_training
        )
      
  def save(self, checkpoint_dir, step):
    model_name = "BranchGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0