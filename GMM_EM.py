import numpy as np
import math
import pickle
import os
import cv2
import matplotlib.pyplot as plt
import sys
from skimage.measure import label,regionprops,find_contours
from Gaussian import Gaussian_likelihood


def load_data(file_name):
    data = open(file_name,'rb')
    data = pickle.load(data)

    return np.matrix(data)




def GMM_model(data, J):

    data=data.T
    n, m = data.shape


    # EM initial paramters
    member_prob = np.matrix(np.ones((n, J))/J)  #membership probability
    mean = np.matrix(np.random.uniform(size = (J, m))*255.)
    cov = np.matrix(np.ones((J,m)) * 1000)
   

    for i in range(80):
      
        last_mean = np.copy(mean)  # deep copy

        # E step
        for j in range(J):
            member_prob[:,j] = Gaussian_likelihood(data, mean[j,:], np.dot(np.diagflat(cov[j,:]),np.eye(3)), 1)
        
        member_prob = np.divide(member_prob, np.sum(member_prob,axis=1)) # normalize each row to be summed up to 1


        # M step
        for j in range(J):
            meb_prob_reshaped = member_prob[:,j]          
            mean[j,:] = np.dot(meb_prob_reshaped.T,data)/np.sum(meb_prob_reshaped,axis=0)
            cov[j,:] = np.dot(meb_prob_reshaped.T, np.square(data-mean[j,:])) / np.sum(meb_prob_reshaped,axis=0) 
            

        print('\n EM iter ', i+1)
        print('  => ',np.sum(np.square(mean-last_mean)))
    

        
        if np.sum(np.square(mean-last_mean))<0.01:
            break

    return mean,cov
    
    

def main():
    
    #train_file_name =['blue_class.pkl','green_class.pkl','brown_class.pkl','not_barrel_blue_class.pkl']
    train_file_name =['blue_class_y.pkl','green_class_y.pkl','brown_class_y.pkl','not_barrel_blue_class_y.pkl']
    #train_file_name =['blue_class_h.pkl','green_class_h.pkl','brown_class_h.pkl','not_barrel_blue_class_h.pkl']
  
    blue_data = load_data(train_file_name[0])
    green_data = load_data(train_file_name[1])
    brown_data = load_data(train_file_name[2])
    not_blue_data = load_data(train_file_name[3])


    print('# of Training samples (Blue,Green,Brown,NotBarrelBlue)')
    print(blue_data.shape[1])
    print(green_data.shape[1])
    print(brown_data.shape[1])
    print(not_blue_data.shape[1])

    
    

    mean_blue,sigma_blue = GMM_model(blue_data,3)
    mean_green,sigma_green = GMM_model(green_data,3)
    mean_brown,sigma_brown = GMM_model(brown_data,3)
    mean_not_blue,sigma_not_blue = GMM_model(not_blue_data,3)
    print(mean_blue)
    print(mean_green)
    print(mean_brown)
    print(mean_not_blue)

    print(sigma_blue)
    print(sigma_green)
    print(sigma_brown)
    print(sigma_not_blue)
    


    '''
    data = open("gmm_model.pkl", "rb")
    data = pickle.load(data)
			
    mean_blue = data['mean_blue']
    sigma_blue = data['sigma_blue']

    mean_green = data['mean_green']
    sigma_green = data['sigma_green']
			
    mean_brown = data['mean_brown']
    sigma_brown = data['sigma_brown']
			
    mean_not_blue = data['mean_not_blue']
    sigma_not_blue = data['sigma_not_blue']
    '''

    

    gmm = {'mean_blue': mean_blue , 'sigma_blue': sigma_blue , 'mean_green': mean_green,
               'sigma_green': sigma_green , 'mean_brown': mean_brown, 'sigma_brown': sigma_brown ,
               'mean_not_blue': mean_not_blue, 'sigma_not_blue': sigma_not_blue}

    pickle.dump(gmm, open("gmm_model.pkl", "wb"))


    


    # load test image
    folder="validationset"
    for filename in os.listdir(folder):
        print(filename)

        test_img = cv2.imread (os.path.join(folder, filename))

        test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2YCR_CB)
    


        h,w,c = test_img.shape
        
        test_data = np.matrix(np.zeros((h*w, 3)))
        test_data[:,0] = np.reshape(test_img[:,:,0],(-1,1))
        test_data[:,1] = np.reshape(test_img[:,:,1],(-1,1))
        test_data[:,2] = np.reshape(test_img[:,:,2],(-1,1))

        Gaussian_blue = Gaussian_likelihood(test_data,mean_blue,sigma_blue,0)
        Gaussian_green = Gaussian_likelihood(test_data,mean_green,sigma_green,0)
        Gaussian_brown = Gaussian_likelihood(test_data,mean_brown,sigma_brown,0)
        Gaussian_not_barrel_blue = Gaussian_likelihood(test_data,mean_not_blue,sigma_not_blue,0)

        #print(Gaussian_blue.shape) 960000x1

        combine = np.hstack((Gaussian_blue,Gaussian_green,Gaussian_brown,Gaussian_not_barrel_blue))

        #print(combine.shape) 960000x4

        max_pdf = np.argmax(combine,axis=1)
        max_pdf = np.array([255 if c==0 else 0 for c in max_pdf])

        # class 0 (blue barrel)                    
        # class 1,2,3(brown, green, not blue barrel)


    
        pred = np.reshape(max_pdf,(800,1200))
        
        prediction = np.zeros((h,w,c))
        cv2.imwrite("mask\\" + filename, pred)

        prediction[:,:,0] = pred
        prediction[:,:,1] = pred
        prediction[:,:,2] = pred

        test_img = cv2.cvtColor(test_img,cv2.COLOR_YCR_CB2BGR)

        bondingbox_result = bonding_box(prediction,test_img)
        



        cv2.imwrite("mask\\bbox_" + filename, bondingbox_result)
        




if __name__ == '__main__':
    main()
