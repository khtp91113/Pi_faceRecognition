# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
import tqdm
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import pickle

def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Get the paths for the corresponding images
            lis = []
            home = os.path.expanduser('~')
            root_folder = home + '/image_align'
            paths = []
            for x in os.listdir(root_folder):
                sub = root_folder + '/' + x
                if x[0] != '.' and os.path.isdir(sub): 
                    sub += '/'
                    paths += [sub + f for f in os.listdir(sub)]
            # Load the model
            model_path = home + '/model'
            facenet.load_model(model_path)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            images_placeholder = tf.image.resize_images(images_placeholder,(160,160))
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            image_size = 160
            embedding_size = embeddings.get_shape()[1]
            extracted_dict = {}
            
            # Run forward pass to calculate embeddings
            for i, filename in enumerate(paths):
                filename = [filename]

                images = facenet.load_data(filename, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                extracted_dict[filename[0]] =  feature_vector

            with open('dict.pickle','wb') as f:
        	    pickle.dump(extracted_dict,f)

if __name__ == '__main__':
    main()
