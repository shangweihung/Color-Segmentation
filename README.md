# Color-Segmentation
Blue Barrel Detection 

# Introduction
![image](https://github.com/shangweihung/Color-Segmentation/blob/master/Image/github.PNG)

Pattern, color and object detection has attracted lots of interest in recent years. The target of this project is to recognize the blue barrel in the images. This kind of object or pattern tracking problem can be seen commonly in recent research such as autonomous driving. The project can be divided into three main tasks. **(1) collecting training samples by ourselves (2) training classifier and predicting (3) post-processing and drawing possible bonding boxes**, as demonstrated in Fig. 1.

We are allowed to implement different machine learning algorithms such as single Gaussian, Gaussian Mixture model or logistic regression. Besides, we can use some built-in functions to evaluate each possible bonding box. By using the training set I have created using roiploy function, I have trained both the **single Gaussian model** and **Gaussian mixture model** to classify each pixels of testing images. Then, the program will perform some image-processing and set some threshold to choose the possible candidate bonding boxes.

# Dependencies
* Python 3    
* opencv-python>=3.4
* matplotlib>=2.2
* numpy>=1.14
* scikit-image>=0.14.
* [roipoly] (https://github.com/jdoepfert/roipoly.py)


# Implementation:
* get_train_sample_***.py: label training sample
* Single_Gaussian.py : single Gaussian Model
* GMM_EM.py :Gaussian Mixture Model
* Barrel_detector.py : Drawing bounding box and Testing 
