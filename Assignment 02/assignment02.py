"""Assignment 02: Fourier Transform
    Name: Julia Marcolan Teixeira 
    Usp Number: 10085370 
    Course code: MAI5020
    Year/semester: 2023.1

Task: The objective of this exercise is to deepen the students understanding of the frequency domain of images and how to use
Fourier Transform and filtering techniques to match the expected results. 
"""
#Importing the necessary libraries. 
import numpy as np 
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def Ideal_Low_pass():
    pass

def Ideal_High_pass():
    pass 

def Ideal_Band_pass():
    pass 

def Laplacian_low_pass():
    pass 

def Gaussian_Low_pass():
    pass 

def Butterworth_low_pass():
    pass 

def Butterworth_high_pass():
    pass 

def RMSE(img_H, img_R): 
    """
        The function calculates the root mean square (RMSE) value between the restored image (img_R) obteined after the filter application 
        and a reference/expectede image (img_H)
        
        Args: 
            - img_H: Expected image. 
            - img_R: Restored image.
        Return: 
            - RMSE_value: root mean square (RMSE) value between the images.
    """

    # gets the size of the input matrix (M x N). Obs: H and img_R should have the same size 
    M, N = img_H.shape

    sum_squares = np.sum(np.square(img_H - img_R))

    RMSE_value = np.sqrt(sum_squares/(M*N))

    return(RMSE_value)

#--------------------------------------------------------------------------------------------------------------------------

#STEP 1: Read the parameters (Input image I, Expected image H, Filter index i ∈ [0,1,2,3,4,5,6], Filter Parameters respective to 
#each index)

#Reading the input image I
filenameI = input().rstrip()
img_I = imageio.imread(filenameI)

#Reading the expected image H
filenameH = input().rstrip()
img_H = imageio.imread(filenameH)


#Reafing the filter index 
index = int(input().rstrip())

#Parameters respecindex == 0: tive to each index

if index == 0: 
    # Ideal Low-pass
    r = int(input().rstrip())
elif index == 0: 
    # Ideal High-pass
    r = int(input().rstrip())
elif index == 2: 
    # Ideal Band-pass
    r1 = int(input().rstrip())
    r2 = int(input().rstrip())
elif index == 3 : 
    #Laplacian low-pass
    pass
elif index == 4: 
    # Gaussian Low-pass
    sigma1 = int(input().rstrip())
    sigma2 = int(input().rstrip())
elif index == 5: 
    # Butterworth low-pass
    D0 = int(input().rstrip())
    N = int(input().rstrip())
elif index == 6: 
    # Butterworth high-pass
    D0 = int(input().rstrip())
    N = int(input().rstrip())
else: 
    print("Index not valid")



#Just  for testing
#plt.figure(figsize=(12,10))
#plt.subplot(121); plt.imshow(img_I, cmap ='gray')
#plt.subplot(122); plt.imshow(img_H, cmap ='gray')
#plt.show()