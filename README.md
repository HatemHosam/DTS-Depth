# DTS-Depth
requirements:<br/>
tensorflow == '2.3.1'<br/>
opencv-python == '4.4.0'<br/>
numpy == '1.18.5'<br/>

In order to run the the test.py, you have to run the code with the following command line:<br/>

python test.py [dataset_name] [path_to_image]<br/>

dataset_name should be one of (KITTI, Cityscapes, or NYUV2)<br/>

example:<br/>
python test.py NYUV2 image.png<br/>

In order to evaluate on a testset, you should run evaluate.py with the following command<br/>

python test.py [dataset_name] [path_to_testset_images_paths.txt]<br/>

example:<br/>
python evaluate.py KITTI KITTI_testset_names.txt<br/>
