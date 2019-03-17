# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# mask rcnn

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("/Users/Fumi/Desktop/Research Project/MRCNN&kalman/Mask_RCNN-master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import videoimage

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
import coco

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

import pandas as pd
import cv2 as cv
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
final_dataframe = pd.DataFrame(columns = ['file_names','classid', 'y1','x1','y2','x2','score','centrex','centrey','filenum'])

for file in os.listdir(ROOT_DIR+'/images/'):
    file_names = ROOT_DIR + '/images/' + file
    print (file_names)
    if file_names.endswith('jpg'):
        filein = open(file_names, "r")
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names))
    # Run detection
        results = model.detect([image], verbose=1)
        print(results)


    # Visualize results & store information in CSV
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
        c = r['rois']
        d = r['class_ids']
        e = r['scores']
        fn = os.path.splitext(os.path.basename(file_names))[0]
        print (fn)
        print (e)
        l = len(d)
        f=np.full((l),int(fn))
        print (f)
        path = ROOT_DIR +'/images/' + str(fn)
        cropimage = cv.imread(file_names)
        cx = np.empty(l)
        cy = np.empty(l)
        filenum = np.empty(l)
        
        i = 0
        ii = 0
        for i in range(0,l):
            cx[i] = (c[i][1]+c[i][3])/2
            cy[i] = (c[i][0]+c[i][2])/2
            print (d[i])
            if d[i] in [3,6,8]:
                if c[i][0] > 200:
                    print (c[i][1])
                    cropImg = cropimage[c[i][0]:c[i][2],c[i][1]:c[i][3]] 
                    filenum[i] = ii
                    ii = ii + 1
            else:
                filenum[i] = -1
            i = i+1
        
        print(f)
        dataframe = pd.DataFrame({'file_names':f,'classid':d, 'y1':c[:,0],'x1':c[:,1],'y2':c[:,2],'x2':c[:,3],'score':e,'centrex':cx,'centrey':cy,'filenum':filenum})
        print (dataframe)
        final_dataframe = pd.merge (final_dataframe, dataframe, on=['file_names','classid', 'y1','x1','y2','x2','score','centrex','centrey','filenum'], how='outer')
    
final_dataframe.to_csv(ROOT_DIR +'/results/result.csv',index=False,sep=',')













