'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.measure import label, regionprops, find_contours
from Gaussian import Gaussian_likelihood


class BarrelDetector():
	def __init__(self):

				
		self.mean_blue = np.matrix([[87.30065819337267, 164.00427541988198, 94.22571777122106]])
		self.sigma_blue = np.matrix([[1215.09627254,   290.00006335, -520.34939405],
        [ 290.00006335,  226.27805361, -231.39182989],
        [-520.34939405, -231.39182989,  370.00547045]])
				
		self.mean_green = np.matrix([[64.03912434167401, 116.87720666350218, 125.99686800009562]])
		self.sigma_green = np.matrix( [[ 8.48335003e+02, -1.40093432e+02, -1.29044073e-01],
        [-1.40093432e+02,  1.01617826e+02, -2.20487040e+01],
        [-1.29044073e-01, -2.20487040e+01,  8.40901475e+01]])
				
		self.mean_brown = np.matrix( [[80.6684737676117, 109.14783147886596, 140.11057477005753]])
		self.sigma_brown = np.matrix([[871.00238841, -12.23194624,  38.10607956],
        [-12.23194624,  47.83237246, -55.50074092],
        [ 38.10607956, -55.50074092,  70.94725317]])
				
		self.mean_not_blue = np.matrix( [[122.15798992237505, 148.7481764946207, 110.1322075446003]])
		self.sigma_not_blue = np.matrix([[940.47524605, -25.19920041, -67.98969579],
        [-25.19920041, 165.48977154, -63.39900663],
        [-67.98969579, -63.39900663, 55.80629786]])

		
		
	def segment_image(self, img):
                # input type is BGR
		img = cv2.cvtColor(img,cv2.COLOR_RGB2YCR_CB)

		h,w,c = img.shape

		test_data = np.matrix(np.zeros((h*w,3)))
		test_data[:,0] = np.reshape(img[:,:,0],(-1,1))
		test_data[:,1] = np.reshape(img[:,:,1],(-1,1))
		test_data[:,2] = np.reshape(img[:,:,2],(-1,1))

		Gaussian_blue = Gaussian_likelihood(test_data,self.mean_blue,self.sigma_blue,1)
		Gaussian_green = Gaussian_likelihood(test_data,self.mean_green,self.sigma_green,1)
		Gaussian_brown = Gaussian_likelihood(test_data,self.mean_brown,self.sigma_brown,1)
		Gaussian_not_barrel_blue = Gaussian_likelihood(test_data,self.mean_not_blue,self.sigma_not_blue,1)

		#Gaussian_blue.shape: 960000x1

		combine = np.hstack((Gaussian_blue,Gaussian_green,Gaussian_brown,Gaussian_not_barrel_blue))
		

		#combine.shape: 960000x4

		max_pdf = np.argmax(combine,axis=1)

		
		result = np.array([1 if c==0 else 0 for c in max_pdf])

		# class 0 (blue barrel)                    
		# class 1,2,3(brown, green, not blue barrel)

		pred = np.reshape(result,(800,1200))
		
		

		


		return pred
        
	def get_bounding_box(self, ori_img):
		mask= self.segment_image(ori_img)
		img = np.zeros((800,1200,3))
		img[:,:,0]=mask
		img[:,:,1]=mask
		img[:,:,2]=mask

		# image processing
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))
		img =cv2.erode(img,kernel)
		img =cv2.dilate(img,kernel,iterations = 3)
		
		
		img = np.array(img,dtype='uint8')
		ret,thresh = cv2.threshold(img, 0, 255, 0)
		thresh = cv2.cvtColor(thresh,cv2.COLOR_RGB2GRAY)
		contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		area_list=[cv2.contourArea(c) for c in contours]
		max_area=contours[area_list.index(max(area_list))]


		
		
		bondbox=[]

		
		if all(np.array(area_list)<1000):
                        
			boxes = []
			
			for c in contours:
			
				area = cv2.contourArea(c)
				if area < max(area_list)/2:
						continue
				
				(x, y, w, h) = cv2.boundingRect(c)
				boxes.append([x,y,x+w,y+h])
				
			boxes = np.asarray(boxes)
			left = np.min(boxes[:,0])
			top = np.min(boxes[:,1])
			right = np.max(boxes[:,2])
			bottom = np.max(boxes[:,3])
			
			bondbox.append([left,top,right,bottom])
			
		else:
			for c in contours:
				area = cv2.contourArea(c)
				if area < max(area_list)/2:
					continue
				
				(x, y, w, h) = cv2.boundingRect(c)
				
				
				if(w/h<=0.4) or (w/h>=1.1):
					continue
				
				bondbox.append([x,y,x+w,y+h])
		bondbox.sort(key=lambda x: x[0])

		
		

		return bondbox
		

if __name__ == '__main__':
    folder = "validation"
    my_detector = BarrelDetector()
	
    for filename in os.listdir(folder):
        print(filename)
					
        # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        
        #cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
        
        
        mask_img = my_detector.segment_image(img)
        
        boxes = my_detector.get_bounding_box(img)
        print(boxes)
        
        
        for i in range(len(boxes)):
            cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2] ,boxes[i][3] ), (0, 0, 255), 2)
        
        cv2.imwrite("ans2\\bbox_" + filename, img)

        #Display results:
        #(1) Segmented images
        #	 mask_img = my_detector.segment_image(img)
        #(2) Barrel bounding box
        #    boxes = my_detector.get_bounding_box(img)
        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
        #Make sure your code runs as expected on the testset before submitting to Gradescope
        
