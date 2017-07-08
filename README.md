#Light CNN, in pytorch
A [pytorch](http://pytorch.org/) implementation of [A Light CNN for Deep Face Representation with Noisy Labels](https://arxiv.org/abs/1511.02683) from the paper by Xiang Wu, Ran He, Zhenan Sun and Tieniu Tan.  The official and original Caffe code can be found [here](https://github.com/AlfredXiangWu/face_verification_experiment).  

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training lightCNN'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#references'>Reference</a>

## Installation
- install [pytorch](http://pytorch.org/) following the website.
- Clone this repository.
	- Note: We currently only run it on Python 2.7.
- Download face dataset such as CASIA-WebFace, VGG-Face and MS-Celeb-1M.
	- The MS-Celeb-1M clean list is uploaded: [Baidu Yun](http://pan.baidu.com/s/1gfxB0iB), [Google Drive](https://drive.google.com/file/d/0ByNaVHFekDPRbFg1YTNiMUxNYXc/view?usp=sharing).

##Datasets
- Download face dataset such as  CASIA-WebFace, VGG-Face and MS-Celeb-1M.
- All face images are converted to gray-scale images and normalized to **144x144** according to landmarks. 
- According to the five facial points, we not only rotate two eye points horizontally but also set the distance between the midpoint of eyes and the midpoint of mouth(ec_mc_y), and the y axis of midpoint of eyes(ec_y) .
- The aligned LFW images are uploaded on [Baidu Yun](https://pan.baidu.com/s/1eR6vHFO).
   Dataset     | size    |  ec_mc_y  | ec_y  
  :----| :-----: | :----:    | :----: 
  Training set | 144x144 |     48    | 48    
  Testing set  | 128x128 |     48    | 40 

##Training lightCNN

