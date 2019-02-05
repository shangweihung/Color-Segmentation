import numpy as np
import os
import pickle
import math

#['BlueData','GreenData','BrownData','notBlueData']
#['blue_class_y.pkl','green_class_y.pkl','brown_class_y.pkl','not_barrel_blue_class_y.pkl']

blue_data=open('blue_class_y.pkl','rb')
blue_data=pickle.load(blue_data)
green_data=open('green_class_y.pkl','rb')
green_data=pickle.load(green_data)
brown_data=open('brown_class_y.pkl','rb')
brown_data=pickle.load(brown_data)
not_blue_data=open('not_barrel_blue_class_y.pkl','rb')
not_blue_data=pickle.load(not_blue_data)


def single_Gaussian_parameters(data):


	data=np.array(data)

	mu_1=np.mean(data[0])
	mu_2=np.mean(data[1])
	mu_3=np.mean(data[2])

	# calculate mean 
	mu_vec=np.array([[mu_1],[mu_2],[mu_3]])

	# calculate covariance
	covar=np.cov(data)
	
	return mu_vec,covar

def main():
        blue_mu,blue_covar = single_Gaussian_parameters(blue_data)
        green_mu,green_covar = single_Gaussian_parameters(green_data)
        brown_mu,brown_covar = single_Gaussian_parameters(brown_data)
        not_blue_mu,not_blue_covar = single_Gaussian_parameters(not_blue_data)


        print(blue_mu,blue_covar)
        print(green_mu,green_covar)
        print(brown_mu,brown_covar)
        print(not_blue_mu,not_blue_covar)




if __name__ == '__main__':
    main()
