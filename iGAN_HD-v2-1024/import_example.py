import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
from im2z import IM2Z
from utils import imread
import scipy
# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
print(latents.shape, labels.shape, latents, labels)
print(Gs)

images, gen_out, kwds, kwds_in = Gs.run(latents, labels)



if False:
  im2z=IM2Z(gen_out, kwds, kwds_in)
  im2z.train()
else:
  im2z=IM2Z(gen_out, kwds, kwds_in)
  im2z.init_states()
  #im2z.test()
  img = imread('./color256.png')[:,:,0:3]/127.5 - 1.
  mask = imread('./mask256.png')[:,:,0:3]/255.
  em = imread('./edge256.png')[:,:,0:3]/127.5 - 1.
  output_size = 1024
  img = scipy.misc.imresize(img, (output_size, output_size))
  mask = scipy.misc.imresize(mask, (output_size, output_size))
  em = scipy.misc.imresize(em, (output_size, output_size))
  print(img.shape)
  #im1, c_err1, e_err1= im2z.predict(color_map=img, name='face256_color')
  im2, c_err2, e_err2= im2z.predict(color_map=img, mask=mask, name='face256_color_mask')
  im3, c_err3, e_err3= im2z.predict(edge_map=em, name='face256_edge')
  im4, c_err4, e_err4= im2z.predict(color_map=img, mask=mask, edge_map = em, name='face256_color_mask_edge')


# Convert images to PIL-compatible format.
#images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
#images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

# Save images as PNG.
#for idx in range(images.shape[0]):
#    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)
