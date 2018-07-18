
# Light CNN for Deep Face Recognition, in pytorch
A [pytorch](http://pytorch.org/) implementation of [A Light CNN for Deep Face Representation with Noisy Labels](https://arxiv.org/abs/1511.02683) from the paper by Xiang Wu, Ran He, Zhenan Sun and Tieniu Tan.  The official and original Caffe code can be found [here](https://github.com/AlfredXiangWu/face_verification_experiment).  

### Table of Contents
- <a href='#updates'>Updates</a>
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training'>Training</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#citation'>Citation</a>
- <a href='#references'>References</a>

## Updates
- Jan 17, 2018
	- Light CNN-29 v2 model and training code are released. The `100% - EER` on LFW achieves **99.43%**.
	- The performance of set 1 on MegaFace achieves **76.021%** for rank-1 accuracy and **89.740%** for TPR@FAR=10^-6. 
- Sep 12, 2017
	- Light CNN-29 model and training code are released. The `100% - EER` on LFW achieves **99.40%**.
	- The performance of set 1 on MegaFace achieves **72.704%** for rank-1 accuracy and **85.891%** for TPR@FAR=10^-6. 
- Jul 12, 2017 
	- Light CNN-9 model and training code are released. The `100% - EER` on LFW obtains **98.70%**.  
	- The performance of set 1 on MegaFace achieves **65.782%** for rank-1 accuracy and **76.288%** for TPR@FAR=10^-6. 
- Jul 4, 2017
	- The repository was built.

## Installation
- Install [pytorch](http://pytorch.org/) following the website.
- Clone this repository.
	- Note: We currently only run it on Python 2.7.	

## Datasets
- Download face dataset such as  CASIA-WebFace, VGG-Face and MS-Celeb-1M.
	- The MS-Celeb-1M clean list is uploaded: [Baidu Yun](http://pan.baidu.com/s/1gfxB0iB), [Google Drive](https://drive.google.com/file/d/0ByNaVHFekDPRbFg1YTNiMUxNYXc/view?usp=sharing).
- All face images are converted to gray-scale images and normalized to **144x144** according to landmarks. 
- According to the five facial points, we not only rotate two eye points horizontally but also set the distance between the midpoint of eyes and the midpoint of mouth(ec_mc_y), and the y axis of midpoint of eyes(ec_y) .
- The aligned LFW images are uploaded on [Baidu Yun](https://pan.baidu.com/s/1eR6vHFO).
  
  Dataset     | size    |  ec_mc_y  | ec_y  
  :----| :-----: | :----:    | :----: 
  Training set | 144x144 |     48    | 48    
  Testing set  | 128x128 |     48    | 40 

## Training 
- To train Light CNN using the train script simply specify the parameters listed in ```train.py``` as a flag or manually change them.
```Shell
python train.py --root_path=/path/to/your/datasets/ \
		--train_list=/path/to/your/train/list.txt \
		--val_list=/path/to/your/val/list.txt \
		--save_path=/path/to/your/save/path/ \
		--model="LightCNN-9/LightCNN-29" --num_classes=n
```
	
- Tips:
	- The lists of train and val datasets are followed by the format of caffe. The details of data loader is shown in ```load_imglist.py```. Or you can use ```torchvision.datasets.ImageFolder``` to load your datasets.
	- The ```num_classes``` denotes the number of identities in your training dataset.
	- When training by pytorch, you can set a larger learning rate than caffe and it is faster converaged by pytorch than caffe for Light CNN.
	- We enlarge the learning rate for the parameters of fc2 which may lead better performance. If the training is collapsed on your own datasets, you can decrese it. 
	- We modify the implementation of SGD with momentum since the official pytorch implementation is different from Sutskever et. al. The details are shown in [here](http://pytorch.org/docs/master/optim.html#torch.optim.SGD).
	- The training datasets for LightCNN-29v2 are CASIA-WebFace and MS-Celeb-1M, therefore, the ```num_classes``` is **80013**. 
	 
	
## Evaluation

- To evaluate a trained network:
```
python extract_features.py --resume=/path/to/your/model \
			   --root_path=/path/to/your/datasets/ \
			   --img_list=/path/to/your/list.txt \
			   --save_path=/path/to/your/save/path/ \
			   --model="LightCNN-9/LightCNN-29/LightCNN-29v2"\
			   --num_classes=n (79077 for LightCNN-9/LightCNN-29, 80013 for LightCNN-29v2)
```
- You can use ```vlfeat``` or ```sklearn``` to evaluate the features on ROC and obtain ```EER``` and ```TPR@FPR``` for your testing datasets. 
- The model of LightCNN-9 is released on [Google Drive](https://drive.google.com/open?id=0ByNaVHFekDPRWk5XUFRvTTRIVmc).
	- Note that the released model contains the whole state of the light CNN module and optimizer. The details of loading model can be found in ```train.py```. 
- The model of LightCNN-29 is released on [Google Drive](https://drive.google.com/file/d/0ByNaVHFekDPRMGlLWVBhbkVGVm8/view).
- The model of LightCNN-29 v2 is released on [Google Drive](https://drive.google.com/open?id=1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS).
- The features of [lfw](https://drive.google.com/open?id=0ByNaVHFekDPRbDV4cEtWSVl3d0k) and [megaface](https://drive.google.com/open?id=0ByNaVHFekDPRZXhQejRwOUtDYm8) of LightCNN-9 are released. 

## Performance
The Light CNN performance on lfw 6,000 pairs.   

|   Model | 100% - EER | TPR@FAR=1%   | TPR@FAR=0.1%| TPR@FAR=0| 
| :------- | :----: | :---: | :---: |:---: | 
| LightCNN-9| 98.70% | 98.47% | 95.13% | 89.53% |
| LightCNN-29 | 99.40% |    99.43%    |    98.67%  |    95.70%  | 
| LightCNN-29v2 | 99.43% |    99.53%    |    99.30%  |    96.77%  | 

The Light CNN performance on [lfw BLUFR protocols](http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/)

|   Model | VR@FAR=0.1% | DIR@FAR=1%| 
| :------- | :----: | :---: |  
| LightCNN-9| 96.80% | 83.06% | 
| LightCNN-29 | 98.95% |    91.33%    |  
| LightCNN-29v2 | 99.41% |    94.43%    |  

The Light CNN performance on MegaFace

|   Model | Rank-1 | TPR@FAR=1e-6| 
| :------- | :----: | :---: |  
| LightCNN-9| 65.782% | 76.288% | 
| LightCNN-29 | 72.704% |  85.891%    |  
| LightCNN-29v2 | 76.021% |  89.740%    |  

## Citation
If you use our models, please cite the following paper:
```
@article{wu2018light,
  title={A light CNN for deep face representation with noisy labels},
  author={Wu, Xiang and He, Ran and Sun, Zhenan and Tan, Tieniu},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={13},
  number={11},
  pages={2884--2896},
  year={2018},
  publisher={IEEE}
}
```
	
## References
- [Original Light CNN implementation (caffe)](https://github.com/AlfredXiangWu/face_verification_experiment).
- [Pytorch](https://github.com/pytorch/pytorch). 


