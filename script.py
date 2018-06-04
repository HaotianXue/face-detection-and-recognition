# This is the script for walking through the whole project.
#
# To run the script, you need to modify the paths in the main function, since the paths are encoded for working on
# a particular machine.  Namely, change all occurance of '/home/ran/Desktop/cvproject/' inside main() to where you
# store the project in your machine. Note that there is no need to modify the helper functions.
#
# Finished on 2018/05/28.

from scipy import misc  
import numpy as np
import tensorflow as tf 
import copy
import cv2
import facenet.facenet
import facenet.align.detect_face
import glob # get list of training image paths
import os

################################# the detection part of the project ########################################
def detect(img_path):
    '''
        Given the path of one test image (the group photo), detect the human faces. 
        (show or save the rectangled detection result image)
        (save the faces detected as single images, with each title is the position of this face in the original image)
        
        str -> void
    '''

    # parameters needed for calling facenet function
    minsize = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709 # scale factor
    gpu_memory_fraction = 1.0
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)  
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))  
        with sess.as_default():  
            pnet, rnet, onet = facenet.align.detect_face.create_mtcnn(sess, None)  
                
    img = misc.imread(img_path)

    bounding_boxes, _ = facenet.align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    # pre-process for handle the possible OutOfBoundError
    for i in range(bounding_boxes.shape[0]):
        if (bounding_boxes[i][0] < 0):
            bounding_boxes[i][0] = 0
        if (bounding_boxes[i][0] > img.shape[1]):
            bounding_boxes[i][0] = img.shape[1]
        if (bounding_boxes[i][1] < 0):
            bounding_boxes[i][1] = 0
        if (bounding_boxes[i][1] > img.shape[0]):
            bounding_boxes[i][1] = img.shape[0]
        if (bounding_boxes[i][2] < 0):
            bounding_boxes[i][2] = 0
        if (bounding_boxes[i][2] > img.shape[1]):
            bounding_boxes[i][2] = img.shape[1]
        if (bounding_boxes[i][3] < 0):
            bounding_boxes[i][3] = 0
        if (bounding_boxes[i][3] > img.shape[0]):
            bounding_boxes[i][3] = img.shape[0]

    crop_faces = [] # list of np.ndarray
    for face_position in bounding_boxes:
        face_position_int = face_position.astype(int)

        # this case will not form a rectangle, so continue
        if (face_position_int[0] == face_position_int[2] or face_position_int[1] == face_position_int[3]):
            continue

        # the sub image of this face
        crop = img[face_position_int[1]:face_position_int[3], face_position_int[0]:face_position_int[2],]  
        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
        crop_faces.append(crop)

        # save this face to disk
        misc.imsave(img_path[:-4] + '/' + str(face_position_int[1]) + '_' + str(face_position_int[0]) + '.jpg', crop)

    # mark rectangle on image
    for face_position in bounding_boxes:
        face_position_int = face_position.astype(int)
        # this case will not form a rectangle, so continue
        if (face_position_int[0] == face_position_int[2] or face_position_int[1] == face_position_int[3]):
            continue
        cv2.rectangle(img, (face_position_int[0], face_position_int[1]), (face_position_int[2], face_position_int[3]), (0, 255, 0), 10) 

    # plt.imshow(img)
    # plt.show()

    # save rectangled detection result image
    misc.imsave(img_path[:-12] + img_path[-8:-4] + '_detection.jpg', img)

################################# the recognition part of the project ########################################


def load_and_align_data(image_paths, image_size=160, margin=44, gpu_memory_fraction=1.0):
    '''
        Given a list of image paths, return a ndarray of all faces.

        [str] -> ndarray
    '''

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = facenet.align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = facenet.align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def knn(test_image, train_images, k=1):
    '''
        Perform KNN on a testing image and training images, find the k most similar training images and their indices.

        ndarray, ndarray, int -> ndarray, [int]
    '''
    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.facenet.load_model('/home/ran/Desktop/cvproject/model_check_point/20180402-114759.pb')

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            train_feed_dict = { images_placeholder: train_images, phase_train_placeholder:False }
            train_emb = sess.run(embeddings, feed_dict=train_feed_dict) # ntrain * 512
            
            test_feed_dict = { images_placeholder: test_image, phase_train_placeholder:False }
            test_emb = sess.run(embeddings, feed_dict=test_feed_dict) # 1 * 512
            
            # Euclidean distances from the test image to all training images
            distances = np.sqrt(np.sum(np.square(test_emb-train_emb), axis=1))
            idx = np.argpartition(distances, k) # get k most similar indices
            return train_images[np.asarray(idx[:k]), :], idx[:k]

def extract_position(path):
    '''
        The cropped faces of a certain test image are titled by the position information (in the detection part).
        This function is to recover it from titles.

        str -> (int, int)
    '''
    i = 42
    y = ""
    while path[i] != '_':
        y = y + path[i]
        i += 1
    
    i += 1
    x = ""
    while path[i] != '.':
        x = x + path[i]
        i += 1
    
    return (int(x), int(y))

if __name__ == "__main__":

    ## DETECTION

    dir_path = os.path.dirname(os.path.realpath(__file__))
    for img_subpath in ['/test/IMG_1818.JPG', '/test/IMG_1819.JPG', '/test/IMG_1820.JPG']:
        detect(dir_path + img_subpath)

    ## RECOGNITION

    # the paths are of random order, but the paths and the images are in consistent order
    train_image_paths = glob.glob("/home/ran/Desktop/cvproject/faces/*")  # [str]
    train_images = load_and_align_data(train_image_paths) # ntrain * 160 * 160 * 3

    for test_subpath in ["IMG_1818", "IMG_1819", "IMG_1820"]:
        test_images_paths = glob.glob("/home/ran/Desktop/cvproject/test/" + test_subpath + "/*") 
        test_images = load_and_align_data(test_images_paths)

        # configuration for putting text on image
        img = misc.imread("/home/ran/Desktop/cvproject/test/" + test_subpath + ".JPG")
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontColor = (255,255,255)
        lineType = 2

        for i in range(test_images.shape[0]):
            test_image = test_images[i, :]
            test_image = test_image.reshape(-1, *test_image.shape) # reshape to add a new dimension, [_, _, _] -> [1, _, _, _]
            similar_image, similar_idx = knn(test_image, train_images, 1)
            label = train_image_paths[similar_idx[0]][-15:-7] # e.g., u5757796

            # put label on image
            position = extract_position(test_images_paths[i])
            cv2.putText(img, label, position, font, fontScale, fontColor, lineType)
        
        misc.imsave("/home/ran/Desktop/cvproject/test/" + test_subpath[-4:] + "_label.jpg", img)
