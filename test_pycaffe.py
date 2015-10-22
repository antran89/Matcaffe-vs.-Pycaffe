__author__ = 'tranlaman'

import numpy as np
import caffe
import time

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
model_dir = '/home/tranlaman/BLVC-caffe/models/bvlc_reference_caffenet/'
net_model = model_dir + 'deploy.prototxt'
net_weights = model_dir + 'bvlc_reference_caffenet.caffemodel'
image_file = '0006.jpg'

import scipy.io as sio
mat = sio.loadmat('test.mat')
input_data = mat['input_data']
input_data = np.transpose(input_data, (3, 2, 0, 1))     # swap channels into [10 x 3 x 227 x 227]

# create net data structures
caffe.set_device(0)  # some errors when setting to GPU 1
caffe.set_mode_gpu()

tic = time.time()
net = caffe.Net(net_model,
                net_weights,
                caffe.TEST)

# set net to batch size of 50
net.blobs['data'].reshape(10,3,227,227)
net.blobs['data'].data[...] = input_data
# forward()
out = net.forward()
toc = time.time()
print("Time elapsed #{}s".format(toc-tic))

# print prediction of center cropping
print("Predicted class of center cropping is #{}.".format(out['prob'][8].argmax()))
print("Confidence of prediction is #{}.".format(out['prob'][8].max()))

scores = np.mean(out['prob'], axis=0)
maxlabel = scores.argmax()
print("Final prediction of ten croppings is #{}.".format(scores.argmax()))
print("Confidence of prediction is #{}.\n".format(scores.max()))

# center cropping from pycaffe preprocessing
mean_array = np.array((104, 117, 123))
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # transpose dimensions to K x H x W
transformer.set_mean('data', mean_array) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

input_data = transformer.preprocess('data', caffe.io.load_image(image_file))
net.blobs['data'].data[...] = input_data
# forward()
out = net.forward()
print("Predicted class of pycaffe preprocess method is #{}.".format(out['prob'][0].argmax()))
print("Confidence of prediction is #{}.".format(out['prob'][0].max()))
