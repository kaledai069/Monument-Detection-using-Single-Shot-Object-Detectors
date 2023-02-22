import glob
import xml.etree.ElementTree as ET
import pandas as pd
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import os
import cv2
import math
import random
import numpy as np
random.seed(10)

ANNOTATION_PATH = 'Annotations'
JPG_IMAGE_PATH = 'JPEGImages'

def saturation(ORI_BASE_DIR, AUG_BASE_DIR, file_name, m_name, grayscale = False):
    # Load image
    image_path = os.path.join(ORI_BASE_DIR, JPG_IMAGE_PATH, file_name + '.jpg')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # saturation_factor of 0.1 equals to grayscale
    # saturation_factor of 2.4 equals to highly saturated images (not highly but perspectively)
    
    if grayscale:
        saturation_factor = 0.0
        s_type = 'grayscale'
    else:
        saturation_factor = 2.25
        s_type = 'saturated'
        
    image = tf.image.adjust_saturation(image, saturation_factor)

    output_file_name = file_name + f'_S_{m_name}_{s_type}'    
    output_image_path = os.path.join(AUG_BASE_DIR, JPG_IMAGE_PATH, output_file_name + '.jpg')
    tf.io.write_file(output_image_path, tf.image.encode_jpeg(image).numpy())

    # input & output annotation path
    input_annotation_path = os.path.join(ORI_BASE_DIR, ANNOTATION_PATH, file_name + '.xml')
    output_annotation_path = os.path.join(AUG_BASE_DIR, ANNOTATION_PATH, output_file_name + '.xml')

    tree = ET.parse(input_annotation_path)
    root = tree.getroot()
    root.find('filename').text = output_file_name
    tree.write(output_annotation_path)

def random_brightness_and_contrast(ORI_BASE_DIR, AUG_BASE_DIR, file_name, m_name, repeat_no = None):
    '''
        Finding
        Brightness
        [0.2, 0.3, 0.4] for brightness on positive side & [-0.1, -0.2] for brightness on negative side
        Contrast
        [-0.85, 1.35] is best upper and lower value (not the range)
    
    '''
    # load image
    image_path = os.path.join(ORI_BASE_DIR, JPG_IMAGE_PATH, file_name + '.jpg')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # set random brightness and random contrast combination
    brightness_range = [0.0, -0.1, -0.2, 0.2, 0.3, 0.4]
    contrast_range = [0.85, 1.35]
    b_value = random.choice(brightness_range)
    c_value = random.choice(contrast_range)
    
    # adjustment to the original image
    image = tf.image.adjust_brightness(image, b_value)
    image = tf.image.adjust_contrast(image, c_value)
    
    if repeat_no is None:
        output_file_name = file_name + f'_BnC_{m_name}'
    else:
        output_file_name = file_name + f'_BnC_{m_name}_{repeat_no}'
    
    # save augmented image
    output_image_path = os.path.join(AUG_BASE_DIR, JPG_IMAGE_PATH, output_file_name + '.jpg')
    tf.io.write_file(output_image_path, tf.image.encode_jpeg(image).numpy())
    

    # copy and save annotation from original to augmented image
    input_annotation_path = os.path.join(ORI_BASE_DIR, ANNOTATION_PATH, file_name + '.xml')
    output_annotation_path = os.path.join(AUG_BASE_DIR, ANNOTATION_PATH, output_file_name + '.xml')
    
    tree = ET.parse(input_annotation_path)
    root = tree.getroot()
    root.find('filename').text = output_file_name
    tree.write(output_annotation_path)

def random_rotation(ORI_BASE_DIR, AUG_BASE_DIR, file_name, m_name, extra = None):
    # Load the image and its annotation
    image_path = os.path.join(ORI_BASE_DIR, JPG_IMAGE_PATH, file_name + '.jpg')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    degree = random.choice(np.concatenate((np.arange(-16, -8, 0.2), np.arange(8, 16, 0.2))))
    angle = degree * math.pi / 180
    image = tfa.image.rotate(image, angle,"bilinear","nearest")

    # Read the annotation file
    input_annotation_path = os.path.join(ORI_BASE_DIR, ANNOTATION_PATH, file_name + '.xml')
    tree = ET.parse(input_annotation_path)
    root = tree.getroot()

    # Get the size of the image
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Perform the rotation on the bounding boxes
    for obj in root.iter("object"):
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        ymin = 512 - ymin
        ymax = 512 - ymax

        # Find the center of the image
        center_x = 256
        center_y = 256

        # Calculate the new bounding box coordinates
        xmin_new = round(center_x + (xmin - center_x) * math.cos(angle) - (ymin - center_y) * math.sin(angle))
        ymin_new = round(center_y + (xmin - center_x) * math.sin(angle) + (ymin - center_y) * math.cos(angle))
        xmax_new = round(center_x + (xmax - center_x) * math.cos(angle) - (ymax - center_y) * math.sin(angle))
        ymax_new = round(center_y + (xmax - center_x) * math.sin(angle) + (ymax - center_y) * math.cos(angle))

        ymin_new = 512 - ymin_new
        ymax_new = 512 - ymax_new

        xmin1 = min([xmin_new,xmax_new])
        xmax1 = max([xmin_new,xmax_new])
        ymin1 = min([ymin_new,ymax_new])
        ymax1 = max([ymin_new,ymax_new])

        # changing y by some suitable value ()
        if(degree < 0):
            xminTemp = xmin + degree
            xmaxTemp = xmax - degree
        else:
            # calculate factors by which bounding box width might increase            
            width_reduce_factor = 0.8 - 0.02 * (degree-10)
            reduced_width = width_reduce_factor * (xmax1-xmin1)
            center_x = (xmax1 + xmin1)/2
            #changing x
            xmin1 = round(center_x - reduced_width/2)
            xmax1 = round(center_x + reduced_width/2)
            xminTemp = min([xmin1,xmax1])
            xmaxTemp = max([xmin1,xmax1])

        #changing y
        reduced_height = (ymax-ymin)-(ymax1-ymin1)
        ymin1 = round(ymin1 - reduced_height/2)
        ymax1 = round(ymax1 + reduced_height/2)
        yminTemp = min([ymin1,ymax1])
        ymaxTemp = max([ymin1,ymax1])

        # checking if final values are between 0 and 300 or 512
        dim_arr = [ xminTemp , yminTemp, xmaxTemp, ymaxTemp ]

        for y in range(0,4):
            i = dim_arr[y]
            i = i if i >= 0 else 0
            i = i if i <= 512 else 512
            dim_arr[y] = round(i)

        xmin1 , ymin1, xmax1, ymax1 = dim_arr

        # Update the bounding box coordinates in the annotation file
        bndbox.find("xmin").text = str(xmin1)
        bndbox.find("ymin").text = str(ymin1)
        bndbox.find("xmax").text = str(xmax1)
        bndbox.find("ymax").text = str(ymax1)

    # Save the augmented image
    if extra is None:
        output_file_name = file_name + f'_RR_{m_name}'
    else:
        output_file_name = file_name + f'_RR_{m_name}_{extra}'
    output_image_path = os.path.join(AUG_BASE_DIR, JPG_IMAGE_PATH, output_file_name + '.jpg')
    tf.io.write_file(output_image_path, tf.image.encode_jpeg(image).numpy())

    # copy the annotation
    output_annotation_path = os.path.join(AUG_BASE_DIR, ANNOTATION_PATH, output_file_name + '.xml')
    root.find('filename').text = output_file_name
    tree.write(output_annotation_path)

def random_translation(ORI_BASE_DIR, AUG_BASE_DIR, file_name, m_name = None, t_type = 0, extra = None):
    # Load the image and its annotation
    image_path = os.path.join(ORI_BASE_DIR, JPG_IMAGE_PATH, file_name + '.jpg')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    dx_dy_list = [[40,0],[0,40],[-40,0],[0,-40],[40,40],[-40,-40],[-40,40],[40,-40]]
    dx_dy = dx_dy_list[t_type]
    # Perform translation augmentation
    image = tfa.image.translate(image, dx_dy,"bilinear","constant")

    # Read the annotation file
    input_annotation_path = os.path.join(ORI_BASE_DIR, ANNOTATION_PATH, file_name + '.xml')
    tree = ET.parse(input_annotation_path)
    root = tree.getroot()

    # Get the size of the image
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Perform the rotation on the bounding boxes
    remove = []
    for obj in root.iter("object"):
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        original_width = xmax-xmin
        original_height = ymax-ymin

        # xmin ymin xmax ymax
        dim_arr = [ xmin + dx_dy[0] , ymin + dx_dy[1] , xmax + dx_dy[0], ymax + dx_dy[1] ] 
        for y in range(0,4):
            i = dim_arr[y]
            i = i if i >= 0 else 0
            i = i if i <= 512 else 512
            dim_arr[y] = i

        xmin1 , ymin1, xmax1, ymax1 = dim_arr

        final_width = xmax1-xmin1
        final_height = ymax1-ymin1

        # if translated bbox does not retain 50% of it's original width and height then remove it
        if( (final_width/original_width) < 0.50 or (final_height/original_height)< 0.50):
            root.remove(obj)
        else:
            # Update the bounding box coordinates in the annotation file
            bndbox.find("xmin").text = str(xmin1)
            bndbox.find("ymin").text = str(ymin1)
            bndbox.find("xmax").text = str(xmax1)
            bndbox.find("ymax").text = str(ymax1)

    # Save the augmented image
    output_file_name = file_name + f'_TRAN_{m_name}_{t_type}'
    output_image_path = os.path.join(AUG_BASE_DIR, JPG_IMAGE_PATH, output_file_name + '.jpg')
    tf.io.write_file(output_image_path, tf.image.encode_jpeg(image).numpy())

    # copy the annotation
    output_annotation_path = os.path.join(AUG_BASE_DIR, ANNOTATION_PATH, output_file_name + '.xml')
    root.find('filename').text = output_file_name
    tree.write(output_annotation_path)

def gaussian_blur(ORI_BASE_DIR, AUG_BASE_DIR, file_name, m_name, repeat_no = None):
    # Load the image and its annotation
    image_path = os.path.join(ORI_BASE_DIR, JPG_IMAGE_PATH, file_name + '.jpg')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    ori_img = image.numpy()

    avg_blur = cv2.blur(ori_img, (5, 5))
    kernels = [(3, 3), (5, 5), (1, 5), (5, 1)]
    kernel = random.choice(kernels)
    
    # applying Gaussian blur with randomly selected kernel size 
    img_gaussian_blur = cv2.GaussianBlur(ori_img, kernel, cv2.BORDER_DEFAULT)
    
    # Save blurred image
    if repeat_no is None:
        output_file_name = file_name + f'_GB_{m_name}'
    else:
        output_file_name = file_name + f'_GB_{m_name}_{repeat_no}'
    output_image_path = os.path.join(AUG_BASE_DIR, JPG_IMAGE_PATH, output_file_name + '.jpg')   
    tf.io.write_file(output_image_path, tf.image.encode_jpeg(img_gaussian_blur).numpy())

    # copy annotation (XML) from the original image XML data
    input_annotation_path = os.path.join(ORI_BASE_DIR, ANNOTATION_PATH, file_name + '.xml')
    tree = ET.parse(input_annotation_path)
    
    output_annotation_path = os.path.join(AUG_BASE_DIR, ANNOTATION_PATH, output_file_name + '.xml')
    root = tree.getroot()
    root.find('filename').text = output_file_name
    tree.write(output_annotation_path)

def gaussian_noise(ORI_BASE_DIR, AUG_BASE_DIR, file_name, m_name = None):
    # Load the image and its annotation
    image_path = os.path.join(ORI_BASE_DIR, JPG_IMAGE_PATH, file_name + '.jpg')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # scaling image in [0,1] because tf.random.normal produces noises between -1 and 1
    image = image.numpy()
    image = np.array(list(map(lambda x : x/255,image)))

    # generating random numbers that follow normal distribution with mean 0.0 and SD = 0.07
    std_dev = float(random.randrange(15, 23)) / float(255)
    noise = tf.random.normal(shape=tf.shape(image), mean = 0.0, stddev = std_dev , dtype=tf.float32)

    #adding the noise to the original image
    noise_img = image + noise
    noise_img = tf.clip_by_value(noise_img, 0.0, 1.0).numpy() 
    noise_img = np.array(list(map(lambda x : x * 255,noise_img)))
    
    # output file name
    output_file_name = file_name + f'_GN_{m_name}'

    #Save noise augmented image
    output_image_path = os.path.join(AUG_BASE_DIR, JPG_IMAGE_PATH, output_file_name + '.jpg')
    tf.io.write_file(output_image_path, tf.image.encode_jpeg(noise_img).numpy())

    # copy the annotation
    input_annotation_path = os.path.join(ORI_BASE_DIR, ANNOTATION_PATH, file_name + '.xml')
    tree = ET.parse(input_annotation_path)
    root = tree.getroot()
    output_annotation_path = os.path.join(AUG_BASE_DIR, ANNOTATION_PATH, output_file_name + '.xml')

    root.find('filename').text = output_file_name
    tree.write(output_annotation_path)


def multiple_orient(ORI_BASE_DIR, AUG_BASE_DIR, file_name, m_name):
    # Load the image and its annotation
    image_path = os.path.join(ORI_BASE_DIR, JPG_IMAGE_PATH, file_name + '.jpg')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Read the annotation file
    input_annotation_path = os.path.join(ORI_BASE_DIR, ANNOTATION_PATH, file_name + '.xml')
    tree = ET.parse(input_annotation_path)
    root = tree.getroot()

    # Get the size of the image
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # Perform the rotation on the image
    orientations = [90, 180]

    for orientation in orientations:
        factor = int(orientation / 90)
        image = tf.image.rot90(image, k=factor)
        angle = math.radians(orientation)

        # Perform the rotation on the bounding boxes
        for obj in root.iter("object"):
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = 512 - int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = 512 - int(bndbox.find("ymax").text)

            # Find the center of the bounding box
            center_x = 256
            center_y = 256

            # Calculate the new bounding box coordinates
            xmin_new = round(center_x + (xmin - center_x) * math.cos(angle) - (ymin - center_y) * math.sin(angle))
            ymin_new = round(center_y + (xmin - center_x) * math.sin(angle) + (ymin - center_y) * math.cos(angle))
            xmax_new = round(center_x + (xmax - center_x) * math.cos(angle) - (ymax - center_y) * math.sin(angle))
            ymax_new = round(center_y + (xmax - center_x) * math.sin(angle) + (ymax - center_y) * math.cos(angle))

            ymin_new = 512 - round(ymin_new)
            ymax_new = 512 - round(ymax_new)

            xmin1 = min([xmin_new,xmax_new])
            xmax1 = max([xmin_new,xmax_new])
            ymin1 = min([ymin_new,ymax_new])
            ymax1 = max([ymin_new,ymax_new])

            dim_arr = [ xmin1 , ymin1, xmax1, ymax1 ] 
            for y in range(0,4):
                i = dim_arr[y]
                i = i if i >= 0 else 0
                i = i if i <= 512 else 512
                dim_arr[y] = i

            xmin1 , ymin1, xmax1, ymax1 = dim_arr
            
            # Update the bounding box coordinates in the annotation file
            bndbox.find("xmin").text = str(xmin1)
            bndbox.find("ymin").text = str(ymin1)
            bndbox.find("xmax").text = str(xmax1)
            bndbox.find("ymax").text = str(ymax1)

        # Save the augmented image
        output_file_name = file_name + f'_MO_{m_name}_{orientation}'
        output_image_path = os.path.join(AUG_BASE_DIR, JPG_IMAGE_PATH, output_file_name + '.jpg')
        tf.io.write_file(output_image_path, tf.image.encode_jpeg(image).numpy())

        # copy the annotation
        output_annotation_path = os.path.join(AUG_BASE_DIR, ANNOTATION_PATH, output_file_name + '.xml')
        root.find('filename').text = output_file_name
        tree.write(output_annotation_path)