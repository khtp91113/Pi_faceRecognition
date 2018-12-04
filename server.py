# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import socket
import threading
import struct
from PIL import Image, ImageDraw, ImageFont
import io
from facenet.src.align import detect_face
from facenet.src import facenet
import pickle
import os
from scipy import misc
import requests
import json
PORT = 20000
BUFFER_SIZE=1024

#load embedding data
with open('/home/k200/dict.pickle','rb') as f:
    feature_array = pickle.load(f) 

#pre-trained model path
model_exp = '/home/k200/model'
graph_fr = tf.Graph()
sess = tf.Session(graph=graph_fr)
image_size = 160
#accurate threshold
threshold = 0.9

#load english to chinese name mapping
with open('/home/k200/name_mapping.txt') as json_file:
    mapping = json.load(json_file)

#load pre-trained model
with graph_fr.as_default():
    saverf = tf.train.import_meta_graph(os.path.join(model_exp, 'model-20180408-102900.meta'))
    saverf.restore(sess, os.path.join(model_exp, 'model-20180408-102900.ckpt-90'))
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

#detect face area in an image
def align_face(img, pnet, rnet, onet):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    if img.size == 0:
        print("empty array")
        return False,img,[0,0,0,0]

    if img.ndim<2:
        print('Unable to align')
        return False,img,[0,0,0,0]

    if img.ndim == 2:
        img = to_rgb(img)

    img = img[:,:,0:3]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    nrof_faces = bounding_boxes.shape[0]
    detect_multiple_faces = True
    margin = 44

    if nrof_faces==0:
        return False,img,[0,0,0,0]
    else:
        det = bounding_boxes[:,0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces>1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2 
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])
        else:
            det_arr.append(np.squeeze(det))
        if len(det_arr)>0:
                faces = []
                bboxes = []
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            faces.append(scaled)
            bboxes.append(bb)
        return True, faces, bboxes

#find out person's face & name, post image on webpage, and return name result
def run_inference_on_image(image):
    images_placeholder = sess.graph.get_tensor_by_name("input:0")
    images_placeholder = tf.image.resize_images(images_placeholder, (160, 160))
    embeddings = sess.graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")

    embedding_size = embeddings.get_shape()[1]
    img = np.array(image)
    #align current image, get faces area & partial image
    response, faces, bboxs = align_face(img, pnet, rnet, onet)

    if response == False:
        post_image(image)
        return 'No face'

    #deal with name result
    names = []
    string = ''
    count = 1
    for face in faces:
        if count != 1:
            string += ','
        images = load_img(face, False, False, image_size)

        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        feature_vector = sess.run(embeddings, feed_dict=feed_dict)

        result = identify(feature_vector, feature_array)
        string += mapping[result]
        names.append(result)
        count += 1
    
    #square face area
    draw = ImageDraw.Draw(image)
    count = 0
    for bound in bboxs:
        left = bound[0]
        up = bound[1]
        right = bound[2]
        down = bound[3]
        draw.line((left, up, right, up), width=3)
        draw.line((left, up, left, down), width=3)
        draw.line((right, up, right, down), width=3)
        draw.line((left, down, right, down), width=3)
        draw.text(((left+right)/2, up), names[count], (255,255,255))
        count += 1
    post_image(image)
    return string

#post current image on webpage
def post_image(image):
    url = 'http://127.0.0.1:5000/put_image'
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}
    try:
        with io.BytesIO() as output:
            image.save(output, format="GIF")
            contents = output.getvalue()
            requests.post(url, data=contents, headers=headers)
    except:
        print('web connection error')

#match face to name by data distance
def identify(feature_vector, feature_array):
    best = np.argsort([np.linalg.norm(feature_vector - pred_row) for ith_row, pred_row in enumerate(feature_array.values())])[0]
    result = feature_array.keys()[best]
    print(np.linalg.norm(feature_vector-feature_array[result]))
    if np.linalg.norm(feature_vector-feature_array[result]) >= threshold:
        return 'unknown'
    return result.split("/")[4]

def main(_):
    server = create_socket()
    run(server)

#recv data from pi, and return result name after face recognization
def tcplink(sock, addr):
    connection = sock.makefile('rb')
    try:
        while True:
            image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
            if not image_len:
                break
            image_stream = io.BytesIO()
            tmp = connection.read(image_len)
            image_stream.write(tmp)
            image_stream.seek(0)
            image = Image.open(image_stream)
            image.verify()
            image = Image.open(image_stream)
            best_result = run_inference_on_image(image).encode('utf-8')
            print(best_result)
            sock.send('%s' % best_result)
    finally:
        sock.close()

def load_img(img, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    images = np.zeros((1, image_size, image_size, 3)) 
    if img.ndim == 2:
        img = facenet.to_rgb(img)
    if do_prewhiten:
        img = facenet.prewhiten(img)
    img = facenet.crop(img, do_random_crop, image_size)
    img = facenet.flip(img, do_random_flip)
    images[:,:,:,:] = img 
    return images

def create_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', PORT))
    s.listen(5)
    print('Waiting for connection...')
    return s

def run(server):
    while True:
        sock, addr = server.accept()
        t = threading.Thread(target = tcplink, args = (sock, addr))
        t.start()

if __name__ == '__main__':
    tf.app.run(main=main)
