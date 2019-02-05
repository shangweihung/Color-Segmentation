import numpy as np
import os
import cv2
import pickle
from roipoly import RoiPoly
import matplotlib.pyplot as plt
from skimage import measure



path = os.listdir('C:\\Users\\Alan\\Desktop\\UCSD\\ECE276A\\ECE276A_HW1\\ECE276A_HW1\\hw1\\trainset_test')


not_barrel_blue_traindata=[[],[],[]]
not_barrel_blue_traindata_yuv=[[],[],[]]
not_barrel_blue_traindata_hsv=[[],[],[]]

for img_name in path:
    
    img=cv2.imread('trainset_test\\'+img_name)
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_yuv=cv2.cvtColor(img,cv2.COLOR_RGB2YCR_CB)
    img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    # show the image
    plt.imshow(img)
    
    
    roi=RoiPoly(color='r')

    # Define ROI of not_barrel_blue barrel
    roi_mask= roi.get_mask(img)

    #plt.imshow(roi_mask)
    #plt.show()


    save_index=np.where(roi_mask==True)

    # pick pixels according to ROI
    channel1=img[:,:,0][save_index]
    channel2=img[:,:,1][save_index]
    channel3=img[:,:,2][save_index]
    
    channel1_yuv=img_yuv[:,:,0][save_index]
    channel2_yuv=img_yuv[:,:,1][save_index]
    channel3_yuv=img_yuv[:,:,2][save_index]

    channel1_hsv=img_hsv[:,:,0][save_index]
    channel2_hsv=img_hsv[:,:,1][save_index]
    channel3_hsv=img_hsv[:,:,2][save_index]

    
    
    for i in range(len(channel1)):
        not_barrel_blue_traindata[0].append(channel1[i])
        not_barrel_blue_traindata[1].append(channel2[i])
        not_barrel_blue_traindata[2].append(channel3[i])
        
        not_barrel_blue_traindata_yuv[0].append(channel1_yuv[i])
        not_barrel_blue_traindata_yuv[1].append(channel2_yuv[i])
        not_barrel_blue_traindata_yuv[2].append(channel3_yuv[i])

        not_barrel_blue_traindata_hsv[0].append(channel1_hsv[i])
        not_barrel_blue_traindata_hsv[1].append(channel2_hsv[i])
        not_barrel_blue_traindata_hsv[2].append(channel3_hsv[i])
                   
# save training data as pickle files

not_barrel_blue_Train_Data=open('not_barrel_blue_class.pkl','wb')
pickle.dump(not_barrel_blue_traindata,not_barrel_blue_Train_Data)
not_barrel_blue_Train_Data.close()

not_barrel_blue_Train_Data_yuv=open('not_barrel_blue_class_y.pkl','wb')
pickle.dump(not_barrel_blue_traindata_yuv,not_barrel_blue_Train_Data_yuv)
not_barrel_blue_Train_Data_yuv.close()

not_barrel_blue_Train_Data_hsv=open('not_barrel_blue_class_h.pkl','wb')
pickle.dump(not_barrel_blue_traindata_hsv,not_barrel_blue_Train_Data_hsv)
not_barrel_blue_Train_Data_hsv.close()

