import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from DTS-Depth_model import DTS-model

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, a1, a2, a3
    
def main(argv):
    #select the weight file (KITTI, Cityscapes, or NYUV2)
    dataset = argv[0]
    if dataset == 'KITTI':
        shape = (224,608,3)
        weights_path = 'KITTI_Weights.h5'
    elif dataset == 'Cityscapes':
        shape = (512,1024,3)
        weights_path = 'Cityscapes_Weights.h5'
    elif dataset == 'NYUV2':
        shape = (480,640,3)
        weights_path = 'NYUV2_Weights.h5'
    else:
        print('invalid weights name')
    
    #load DTS-Depth model with the selected pretrained weights
    model = DTS-model(shape)
    model.load_Weights(weights_path)
    #read test image
    imgs = []
    depths = []
    files = []

    images_path = './images/'
    depth_path = './depth_gt/'

    with open("test.txt", "r") as f:
        val_data = f.readlines()
    for data in val_data:
        files.append(data)

    for file in files:
        #save start time to be used to calculate processing time
        start = time.time()
        img = cv2.imread(images_path+file.split('\n')[0]+'.png')
        
        if dataset == 'KITTI':
            h,w,_ = img.shape
            if h == 370:
                img = img[146:,:]
            else:
                img = img[151:,:]
            depth_gt = np.array(cv2.imread(file.split('\n')[0]+'png', cv2.IMREAD_UNCHANGED)/ 256., dtype = np.uint8)
            
            depth_gt = cv2.resize(depth_png, dsize=(shape[1] , shape[0]), interpolation=cv2.INTER_NEAREST)
        elif dataset == 'Cityscapes':
            depth_gt= cv2.imread(file.split('\n')[0]+'.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_gt = cv2.resize(depth_png, dsize=(shape[1] , shape[0]), interpolation=cv2.INTER_NEAREST)
            depth_gt[depth_gt > 0] = (depth_gt[depth_gt > 0] - 1) / 256
        elif dataset == 'NYUV2':
            depth_gt = np.load(depth_path+file.split('\n')[0]+'.npy')*10.0
        
        gt = depth_gt
        
        img = cv2.resize(img, (shape[1] , shape[0]))    
        img = img / 255.
        img= np.expand_dims(img, axis=0)
        pred = model.predict(img)
        now = time.time()
        print(now - start)  #show processing time
        pred = pred[0,:,:,0]
        #exclude the invalid depth values in evaluation
        ind = np.where(depth == 0)
        gt[gt == 0] = 1
        pred[ind] = 1
        #append the error value
        errors.append(compute_errors(gt, pred))
    
    mean_errors = np.array(errors).mean(0)  
    print(mean_errors)
