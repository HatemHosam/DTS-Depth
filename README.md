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
python evaluate.py KITTI testset_names.txt<br/>

Note: the ground truth depth were extracted from the .mat file and saved in .npy format for easier and faster loading during training of the models.

If you use this code, please cite this paper:
H. Ibrahem, A. Salem, and H.-S. Kang, “DTS-Depth: Real-Time Single-Image Depth Estimation Using Depth-to-Space Image Construction,” Sensors, vol. 22, no. 5, p. 1914, Mar. 2022, doi: 10.3390/s22051914.
