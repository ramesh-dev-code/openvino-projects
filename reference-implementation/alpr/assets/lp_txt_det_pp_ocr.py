# ==============================================================================
# Copyright (C) 2018-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

from gstgva import VideoFrame
import cv2
import sys
import json

import os
import numpy as np
import time
from pathlib import Path
from openvino.runtime import Core
import copy

import paddle
import math
import collections
from assets import pre_post_processing as processing

a = 0
ie = Core()
# Text Detection Model
det_compiled_model = ie.compile_model(model='models/intel/horizontal-text-detection-0001/FP16-INT8/horizontal-text-detection-0001.xml', device_name="CPU")
det_input_layer = det_compiled_model.input(0)
# Text Recognition Model
rec_model = ie.read_model(model='models/en_PP-OCRv3_rec_infer/inference.pdmodel')
# Assign dynamic shapes to every input layer on the last dimension.
for input_layer in rec_model.inputs:
    input_shape = input_layer.partial_shape
    input_shape[3] = -1
    rec_model.reshape({input_layer: input_shape})

rec_compiled_model = ie.compile_model(model=rec_model, device_name="CPU")
# Get input and output nodes.
rec_input_layer = rec_compiled_model.input(0)
rec_output_layer = rec_compiled_model.output(0)

# Preprocess for text recognition.
def resize_norm_img(img, max_wh_ratio):
    """
    Resize input image for text recognition

    Parameters:
        img: bounding box image from text detection 
        max_wh_ratio: value for the resizing for text recognition model
    """
    rec_image_shape = [3, 48, 320]
    imgC, imgH, imgW = rec_image_shape
    assert imgC == img.shape[2]
    character_type = "ch"
    if character_type == "ch":
        imgW = int((32 * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def multiply_by_ratio(ratio_x, ratio_y, box):
    return [
        max(shape * ratio_y, 10) if idx % 2 else shape * ratio_x
        for idx, shape in enumerate(box[:-1])
    ]

def get_rotate_crop_image(img, bb_loc):    
    (xmin, ymin, xmax, ymax) = map (int, bb_loc)
    img_crop_width = xmax - xmin
    img_crop_height = ymax - ymin   

    points = np.float32([[xmin,ymin], [xmax,ymin], [xmax,ymax],[xmin,ymax]])   

    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                            [img_crop_width, img_crop_height],
                            [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def prep_for_rec(dt_boxes, frame):
    """
    Preprocessing of the detected bounding boxes for text recognition

    Parameters:
        dt_boxes: detected bounding boxes from text detection 
        frame: original input frame 
    """
    ori_im = frame.copy()
    img_crop_list = [] 
    
    
    tmp_box = copy.deepcopy(dt_boxes)
    img_crop = get_rotate_crop_image(ori_im, tmp_box)
    img_crop_list.append(img_crop)
        
    img_num = len(img_crop_list)
    # Calculate the aspect ratio of all text bars.
    width_list = []
    for img in img_crop_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    
    # Sorting can speed up the recognition process.
    indices = np.argsort(np.array(width_list))
    return img_crop_list, img_num, indices


def batch_text_box(img_crop_list, img_num, indices, beg_img_no, batch_num):
    """
    Batch for text recognition

    Parameters:
        img_crop_list: processed detected bounding box images 
        img_num: number of bounding boxes from text detection
        indices: sorting for bounding boxes to speed up text recognition
        beg_img_no: the beginning number of bounding boxes for each batch of text recognition inference
        batch_num: number of images for each batch
    """
    norm_img_batch = []
    max_wh_ratio = 0
    end_img_no = min(img_num, beg_img_no + batch_num)
    for ino in range(beg_img_no, end_img_no):
        h, w = img_crop_list[indices[ino]].shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    for ino in range(beg_img_no, end_img_no):
        norm_img = resize_norm_img(img_crop_list[indices[ino]], max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)

    norm_img_batch = np.concatenate(norm_img_batch)
    norm_img_batch = norm_img_batch.copy()
    return norm_img_batch

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [N, 5]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[1], x[0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][1] - _boxes[i][1]) < 10 and \
                (_boxes[i + 1][0] < _boxes[i][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def txt_det_ocr(img):    
    global det_compiled_model
    global det_input_layer
    global rec_compiled_model
    global rec_input_layer
    global rec_output_layer
    
    # Text detection
    N, C, H, W = det_input_layer.shape
    # Resize the image to meet network expected input sizes.
    resized_image = cv2.resize(img, (W, H))
    # Reshape to the network input shape
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    #ts1 = time.time()
    output_key = det_compiled_model.output("boxes")    
    #print('Text Detection Inference Time: {} ms'.format((time.time()-ts1)*1000))
    boxes = det_compiled_model([input_image])[output_key]    
    # Remove zero only boxes.
    dt_boxes = boxes[~np.all(boxes == 0, axis=1)]    
    #print('Text det results: ', dt_boxes) 
    ts1 = time.time()
    ocr_res = ''
    if len(dt_boxes) > 0:         
        dt_boxes = sorted_boxes(dt_boxes)        
        for i in range(len(dt_boxes)):
            box = dt_boxes[i]
            (real_y, real_x), (resized_y, resized_x) = img.shape[:2], resized_image.shape[:2]
            ratio_x, ratio_y = real_x / resized_x, real_y / resized_y     
            
            # Pick a confidence factor from the last place in an array.
            conf = box[-1]                 
            if conf > 0.3:
                # Convert float to int and multiply corner position of each box by x and y ratio.
                # If the bounding box is found at the top of the image, 
                # position the upper box bar little lower to make it visible on the image. 
                (x_min, y_min, x_max, y_max) = [
                    int(max(corner_position * ratio_y, 10)) if idx % 2 
                    else int(corner_position * ratio_x)
                    for idx, corner_position in enumerate(box[:-1])
                ]
                #print('BB Coord: ',x_min, y_min, x_max, y_max)
                '''
                img_crop = get_rotate_crop_image(frame, [x_min, y_min, x_max, y_max])                
                cv2.imshow('Crop', img_crop)    
                cv2.waitKey(0)                       
                '''           
                img_crop_list, img_num, indices = prep_for_rec([x_min, y_min, x_max, y_max],img)                         
                batch_num = img_num
                
                # For storing recognition results, include two parts:
                # txts are the recognized text results, scores are the recognition confidence level. 
                rec_res = [['', 0.0]] * img_num
                txts = [] 
                scores = []            

                for beg_img_no in range(0, img_num, batch_num):

                    # Recognition starts from here.
                    norm_img_batch = batch_text_box(
                        img_crop_list, img_num, indices, beg_img_no, batch_num)
                    #ts2 = time.time()
                    # Run inference for text recognition. 
                    rec_results = rec_compiled_model([norm_img_batch])[rec_output_layer]
                    #print('OCR Inf Time: {} ms'.format((time.time() - ts2)*1000))
                    # Postprocessing recognition results.
                    postprocess_op = processing.build_post_process(processing.postprocess_params)
                    rec_result = postprocess_op(rec_results)                                                
                            
                    for rno in range(len(rec_result)):
                        rec_res[indices[beg_img_no + rno]] = rec_result[rno]                  
                    if rec_res:
                        txts = [rec_res[i][0] for i in range(len(rec_res))]                    
                        scores = [rec_res[i][1] for i in range(len(rec_res))]
                        sc_ind = [ ind for ind, val in enumerate(scores) if val > 0.5]                                                    
                        for i in range(len(sc_ind)):
                            ocr_res += str(txts[i])                    
                        #print(ocr_res)    
                                    
        #if ocr_res:                
            #print('OCR Latency: {} ms'.format((time.time()-ts1)*1000))
    #else:
        #print('No text is detected')
    return ocr_res     

def process_frame(frame: VideoFrame) -> bool: 
    global a      
    messages = list(frame.messages())    
    mes_len = len(messages)    
    if len(messages) > 0:        
        frame.remove_message(messages[0])
    
    with frame.data() as mat:                
        a += 1
        for roi in frame.regions():                                
            for tensor in roi.tensors():                        
                if roi.label()=='license-plate':             
                    lp_box = roi.rect()                                        
                    #gray = cv2.cvtColor(mat[lp_box.y-15:lp_box.y+lp_box.h+15,lp_box.x-20:lp_box.x+lp_box.w+20], cv2.COLOR_BGR2GRAY)
                    gray = mat[lp_box.y-30:lp_box.y+lp_box.h+30,lp_box.x-30:lp_box.x+lp_box.w+30,0:3]
                    cv2.imwrite('./out_img/lp_{}.jpg'.format(a),gray)
                    st_ts = time.time()                                 
                    lp_num = txt_det_ocr(gray)
                    if len(lp_num) > 4:  
                        print('Text Detection & Recognition Time: {} ms'.format((time.time() - st_ts)*1000))
                        print('LP Number: ',lp_num+'\n')                  
                        # Add a new field 'LP' with the value of lp_num to the existing frame message 
                        if len(messages) > 0:         
                            json_msg = json.loads(messages[0])
                            obj = json_msg['objects']               
                            for i in range(len(obj)):
                                if obj[i]['detection']['label'] == 'license-plate':                
                                    obj[i]['detection']['label_id'] = lp_num
                                if obj[i]['detection']['label'] == 'vehicle':                
                                    del obj[i]['detection']
                            json_msg['objects'] = obj                
                            messages[0] = json.dumps(json_msg)
                            frame.add_message(messages[0])
                        cv2.putText(mat,lp_num,(lp_box.x, lp_box.y),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 1)
                        #cv2.imwrite('demo_img/demo/res_{}.jpg'.format(a),mat)                                                
                            
    return True
