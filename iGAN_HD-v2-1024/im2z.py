from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from tensorflow.python.ops import data_flow_ops


from ops import *
from utils import *
from tf_hog import *
from optimizer import AdamOptimizer

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))


class IM2Z(object):
  def __init__(self, gen_out, kwds, kwds_in):

    self.sess = tf.get_default_session()
    self.gen_out = tf.transpose(gen_out[0], [0,2,3,1])

    #self.branchgan = branchgan
    self.crop = False

    self.batch_size, self.output_height, self.output_width, self.c_dim = self.gen_out.get_shape().as_list()
    self.batch_size = 1
    print(self.batch_size, self.output_height, self.output_width, self.c_dim)

    #self.output_height = flags.output_height
    #self.output_width = flags.output_width
    self.dataset_name = 'celeba_hq'
    self.input_fname_pattern = '*.jpg'

    height_pyramid = [self.output_height]
    width_pyramid = [self.output_width]
    while height_pyramid[-1]>8 and width_pyramid[-1]>8:
      height_pyramid.append(conv_out_size_same(height_pyramid[-1], 2))
      width_pyramid.append(conv_out_size_same(width_pyramid[-1], 2))
    self.height_pyramid, self.width_pyramid = list(reversed(height_pyramid)), list(reversed(width_pyramid))
    print('----height_pyramid/width_pyramid: ', height_pyramid, width_pyramid)
    self.n_levels = len(height_pyramid)

    self.z_dim = kwds_in[0].shape[1]
    self.z = kwds[0]
    self.clss = kwds[1]

    self.clss_in = kwds_in[1][0:1]

    self.df_dim = 64
    self.gf_dim = 64
    self.epoch = 15

    self.use_z_pyramid = False
    self.use_residual_block = False

    self.checkpoint_dir = "checkpoint"
    self.use_two_stage_training = False

    self.build_model()

  def init_states(self):
    init_new_vars_op = tf.initialize_variables(self.vars_z_net + self.optimizer_vars)
    self.sess.run(init_new_vars_op)
    #try:
    #  tf.global_variables_initializer().run()
    #except:
    #  tf.initialize_all_variables().run()

    
    could_load, counter_load = self.load(self.checkpoint_dir)
    if could_load:
      counter = counter_load
      print(" [*] im2z Load SUCCESS")
    else:
      counter = 1
      print(" [!] im2z Load failed...")

    self.build_z_opt_net()
  
  def build_z_opt_net(self):


    self.color_map_placeholder = tf.placeholder(shape=[1,self.output_height,self.output_width,self.c_dim], dtype=tf.float32)
    self.mask_placeholder = tf.placeholder(shape=[1,self.output_height,self.output_width,self.c_dim], dtype=tf.float32)
    self.edge_map_placeholder = tf.placeholder(shape=[1,self.output_height,self.output_width,self.c_dim], dtype=tf.float32)
    self.c_ratio_placeholder = tf.placeholder(dtype=tf.float32)
    self.e_ratio_placeholder = tf.placeholder(dtype=tf.float32)

    self.loss_c = tf.reduce_sum(tf.abs(self.color_map_placeholder - self.gen_out)* \
      self.mask_placeholder)/ (tf.reduce_sum( self.mask_placeholder ) + 10.)
    self.loss_e = self.edge_loss(self.gen_out,  self.edge_map_placeholder)

    self.loss_best = self.loss_e * self.e_ratio_placeholder + self.loss_c * self.c_ratio_placeholder

    self.z_gradients_predict = tf.gradients(self.loss_best, self.z)[0]
    
    #self.clip_z_op = self.z_best.assign(tf.clip_by_value(self.z_best, -0.8, 0.8))

  
  def build_model(self):

    self.image_batch_placeholder = tf.placeholder(tf.float32, \
      shape=(self.batch_size, self.output_height, self.output_width, self.c_dim), name='input_image')

    self.z_predict = self.z_net(self.image_batch_placeholder)

    self.vars_z_net = [var for var in tf.trainable_variables() if 'z_net' in var.name]

    self.loss = tf.reduce_mean(tf.abs(self.gen_out - self.image_batch_placeholder))

    self.z_gradients_train = tf.gradients(self.loss, self.z) #self.flags.beta1
    self.z_grads_placeholder = tf.placeholder(tf.float32, \
      shape=(self.batch_size, self.z_dim), name='z_grads_placeholder')
    opt = AdamOptimizer(alpha=0.0001, beta1=0.9, loss = self.loss, t_vars = self.vars_z_net)
    self.train_op = opt.update(self.z_predict, self.z_grads_placeholder)

    self.optimizer_vars = opt.get_variables()#[var for var in tf.GraphKeys.GLOBAL_VARIABLES() if 'optimizer' in var.name]
    for var in self.optimizer_vars:
      print(var)
    self.saver = tf.train.Saver()


  def edge_loss(self, G, edge_map):
    G = tf.reshape(G, edge_map.get_shape().as_list())
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
    fdict[self.clss] = self.clss_in
    
    #self.sess.run(self.z_best.initialized)
    #self.sess.run(self.z.assign(z0))
    value_o, value = -4, 4
    z_value = z0
    while np.abs(value - value_o) > 1e-4: #czq_flag: remember to change back to 1e-4
      #czq_flag: if you want to skip some images, say sample 1 out of 4, add code here
      for i in range(4):
        value_o = value
        fdict[self.z]  = z_value
        value, z_grads = self.sess.run([self.loss_best, self.z_gradients_predict], feed_dict=fdict)
        #print(z_grads.shape)
        z_value -= 0.1 * z_grads
        z_value = np.clip(z_value, -0.8, 0.8)
        print(value)
      fdict[self.z]  = z_value
      img = self.sess.run(self.gen_out, feed_dict=fdict)
      img = ((img[0]+1)*127.999).astype(np.uint8)
      signal_function(img)
    print('[End] optimizing z, time spent ' + '{}'.format(time.time()-time_start))
    #return img_best, err_c, err_e

  def predict(self, signal_function, color_map = None, mask = None, edge_map=None, name='output'):
    err_c, err_e= 0.0, 0.0
    if color_map is not None and mask is None:
      image = np.expand_dims(color_map, axis=0)
      z0 = self.sess.run(self.z_predict, \
        feed_dict={self.image_batch_placeholder:image})
    else:
      if self.use_z_pyramid:
        z0 = np.random.uniform(-1, 1, [1, self.z_dim]).astype(np.float32)
      else:
        z0 = np.random.uniform(-1, 1, [1, self.z_dim]).astype(np.float32)
    self.find_best(signal_function, z0, image=color_map, mask=mask, edge_map=edge_map)
    #print(out_img.shape)
    #scipy.misc.imsave('./tests/'+name+'.png', (out_img[0]+1.)*2.)
    #print('[SUCCESS] err_c: {}  err_e: {}'.format(err_c, err_e))
    #print('[SUCCESS] output image saved to '+name+'.png')


  def train(self):
    start_time = time.time()
    
    init_new_vars_op = tf.initialize_variables(self.vars_z_net + self.optimizer_vars)
    self.sess.run(init_new_vars_op)


    could_load, counter_load = self.load(self.checkpoint_dir)
    if could_load:
      counter = counter_load
      print(" [*] im2z Load SUCCESS")
    else:
      counter = 1
      print(" [!] im2z Load failed...")

    for epoch in xrange(self.epoch):
      batch_idxs = 5000

      self.data = glob(os.path.join(
          "./data", self.dataset_name, self.input_fname_pattern))
      np.random.shuffle(self.data)
      self.data = np.array(self.data)
      batch_idxs = len(self.data) // self.batch_size

      for idx in xrange(0, batch_idxs):
          batch_imgs = np.asarray([get_image(self.data[i], self.output_height, self.output_width, crop=False) 
            for i in range(idx * self.batch_size ,(idx+1) * self.batch_size, 1)])

          z_predict = self.sess.run(self.z_predict, feed_dict={self.image_batch_placeholder:batch_imgs})
          z_grads, loss = self.sess.run([self.z_gradients_train, self.loss], 
            feed_dict = {self.z: z_predict, self.clss:self.clss_in, self.image_batch_placeholder:batch_imgs})
          #print(z_grads)
          __ = self.sess.run(self.train_op, feed_dict={ self.z_grads_placeholder:z_grads[0], 
            self.image_batch_placeholder:batch_imgs})
          ##z_net_gradients = self.sess.run(self.z_net_gradients)
          

          counter += 1
          print("epoch=%2d batch_nd=%4d/%4d time=%4.4f, loss=%.8f" \
            % (epoch, idx, batch_idxs, time.time() - start_time, loss))

          if np.mod(counter, 500) == 2:
            self.save(self.checkpoint_dir, counter)

  
  def z_net(self, inputs, reuse = False):
      if self.use_z_pyramid:
        z_concat = self.sub_discriminator(inputs, self.z_dim * (self.n_levels-1), reuse=reuse)
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
      for level in range(self.n_levels - 1, 0, -1):
          height, width = self.height_pyramid[level], self.width_pyramid[level]
          df_dim = self.get_f_dim(self.df_dim, height, width)
          print(h, df_dim)
          h = conv2d(h, df_dim, name='znet_h_'+str(level))
          #h = self.residual_block(h, level, name='znet_')
          bn = instance_batch_norm(name='znet_bn_'+str(level))
          h = lrelu(bn(h))
      z = linear(tf.reshape(h, [inputs.get_shape()[0], -1]), dims, 'znet_lin_'+str(self.n_levels - 1))              
      return z



  def test(self):

    if self.use_z_pyramid:
        batch_z = []
        batch_z.append(np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32))
        z_dict = {self.z_pyramid[0]:batch_z[0]}
        for level_tmp in range(1, self.n_levels-1, 1):
          batch_z.append(np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32))
          z_dict [self.z_pyramid[level_tmp]] = batch_z[level_tmp]
    else:
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        z_dict = {self.z: batch_z}

    z_dict[self.clss] = self.clss_in
    imgs = self.sess.run(self.gen_out, feed_dict=z_dict)
    loss, imgs_recover = self.sess.run([self.loss, self.imgs_recover], feed_dict={self.image_batch_placeholder:imgs})
    print(imgs_recover.shape)
    save_images(imgs, image_manifold_size(self.batch_size), './tests/test_img.png')
    save_images(imgs_recover, image_manifold_size(self.batch_size), './tests/test_img_recover.png')

    data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
    imgs_raw = np.array([get_image(path, self.output_height, self.output_width, crop=False) 
    	for path in data[0:self.batch_size]])
    print(imgs_raw.shape)
    loss, imgs_recover = self.sess.run([self.loss, self.imgs_recover], 
    	feed_dict={self.image_batch_placeholder:imgs_raw}) 
    print(imgs_recover.shape)
    save_images(imgs_raw, image_manifold_size(self.batch_size), './tests/test_raw_imgs.png')
    save_images(imgs_recover, image_manifold_size(self.batch_size), './tests/test_raw_imgs_recover.png')
    
    for i in range(self.batch_size):
      img_recover_n, __, __ = self.predict(color_map = imgs_raw[i]) 
      imgs_recover[i] = img_recover_n[0]
    save_images(imgs_raw, image_manifold_size(self.batch_size), './tests/test_raw_imgs_bp.png')
    save_images(imgs_recover, image_manifold_size(self.batch_size), './tests/test_raw_imgs_recover_bp.png')
    

  
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
        self.dataset_name, self.n_levels, 
        self.output_height, self.output_width,
        self.use_z_pyramid, self.z_dim,
        self.df_dim, self.gf_dim, self.use_residual_block,
        self.batch_size, self.epoch, self.use_two_stage_training
        )
      
  def save(self, checkpoint_dir, step):
    model_name = "im2z.model"
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