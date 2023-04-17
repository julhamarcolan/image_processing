"""Assignment 02: Fourier Transform
    Name: Julia Marcolan Teixeira 
    Usp Number: 10085370 
    Course code: MAI5020
    Year/semester: 2023.1

Task: The objective of this exercise is to deepen the students understanding of the frequency domain of images and how to use
Fourier Transform and filtering techniques to match the expected results. 
"""
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

def Butterworth_band_reject():
    pass 

def Butterworth_band_pass():
    pass 

def RMSE():
    pass 
#--------------------------------------------------------------------------------------------------------------------------
#Importing the necessary libraries. 
import numpy as np 
import imageio.v2 as imageio
import matplotlib.pyplot as plt

#STEP 1: Read the parameters (Input image I, Expected image H, Filter index i ∈ [0,1,2,3,4,5,6,7,8], Filter Parameters respective to 
#each index)

#Reading the input image I
filenameI = input().rstrip()
img_I = imageio.imread(filenameI)

#Reading the expected image H
filenameH = input().rstrip()
img_H = imageio.imread(filenameH)

#Reafing the filter index 
index = input().rstrip()

#Parameters respective to each index
if index == 0: 
    r = input().rstrip()
elif index == 1: 
    r = input().rstrip()
elif index == 2: 
    r1 = input().rstrip()
    r2 = input().rstrip()
elif index == 3: 
    pass 
elif index == 4: 
    sigma1 = input().rstrip()
    sigma2 = input().rstrip()
elif index == 5: 
    D0 = input().rstrip()
    N = input().rstrip()
elif index == 6: 
    D0 = input().rstrip()
    N = input().rstrip()
elif index == 7: 
    D0 = input().rstrip()
    N = input().rstrip()   
elif index == 8: 
    D0 = input().rstrip()
    D1 = input().rstrip()
    N1 = input().rstrip()
    N2 = input().rstrip()
