import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from DTS-Depth_model import DTS-model

def main(argv):
    #select the weight file (KITTI, Cityscapes, or NYUV2)
    if argv[0] == 'KITTI':
        shape = (224,608,3)
        weights_path = 'weights/KITTI_Weights.h5'
    elif argv[0] == 'Cityscapes':
        shape = (512,1024,3)
        weights_path = 'weights/Cityscapes_Weights.h5'
    elif argv[0] == 'NYUV2':
        shape = (480,640,3)
        weights_path = 'weights/NYUV2_Weights.h5'
    else:
        print('invalid weights name')
    
    #load DTS-Depth model with the selected pretrained weights
    model = DTS-model(shape)
    model.load_Weights(weights_path)
    #read test image
    img_path = argv[1]
    image = cv2.imread(img_path)
    image = cv2.resize(image, (shape[1] , shape[0]))
    img = image/255.
    img = np.expand_dims(img, axis = 0)
    #Depth estimation using the loaded weights
    depth_map = model.predict(img)
    depth_map = depth_map[0,:,:,0]
    
    plt.figure('predicted depth-map')
    plt.imshow(depth_map, cmap= 'magma')
    plt.show()