"""Assignment 01:  image descriptors
    Name: Julia Marcolan Teixeira 
    Usp Number: 10085370 
    Course code: MAI5021
    Year/semester: 2023.1

Task: In this assignment you have to implement a classic feature extractor called Histogram of Oriented Gradients (HoG) and 
implement and Train a K-nearest-neighbours. 
"""
def read_image_collection(filename):
    """
        Reads an image collection from a text file containing the image file names.
        
        Args:
            - filename: Text file containing the image file names.
        
        Returns:
            - images: List of images in the collection.
    """
    names = filename.split()
    images = []
    for name in names:
        images.append(imageio.imread(name))
    
    return images

def Luminance(Img):
    """
        The Luminance function is used to obtain a single channel from the RGB image, combining the colour channels using a
        RGB to grayscale conversion. The method is a weighted averabe of the colour channels, using the weighs: 
        R ->  0.299 
        G ->  0.587
        B -> 0.114

        Args: 
            - Img: image RGB output.
        Return: 
            - ImgG: image output in grey scale. 
    """
    #Making a copy of the image using float values to avoid errors
    Img = np.array(Img, copy=True).astype(float)
    M = Img.shape[0]
    N = Img.shape[1]

    # create the image to store the new  image
    new_img = np.zeros((M, N))

    #calculating the new image using the weighted averabe
    new_img = Img[:,:,0]*0.299 + Img[:,:,1]*0.587 + Img[:,:,2]*0.114

    #normalizing the new image 
    f_min = np.min(new_img)
    f_max = np.max(new_img)
    new_img = ((new_img - f_min)/(f_max-f_min))*255

    return(new_img)

def HoG(Img):
    """
        The HoG function computes the Histogram of Oriented Gradients descripor for an image. The gradient is obtained
        by combining magnitude and angle from the image. 
        
        The magnitude are calculated by: 
       
        Mxy(i,j) = np.sqrt(gx(i,j)**2 + gy(i,j)**2)/ sum_jj,
        where sum_ij is the sum for all is and all j in sqrt(gx(i,j)**2 + gy(i,j)**2)

        The angles are calculated by: 
        Phixy(i,j) = np.arctan(gy(i,j)/gx(i,j)).

        DETERMINING IN WHICH OF THE 9 BINS THE VALUE FALLS INTO: 
        In order to obtain the descriptro, the angles will be digitese into 9 bins, this means slicing the angle range in 20 degree
        intervals [0, 19], [20, 39]. To do this, we are going to divide the angles matrix for 20 and use the integer part of the result
        to classify. Ex: if we divide 10/20 the result will be 0.5 that menas that the angle 10 is stores in the frist interal [0,19]. 
        This implementation will avoid use a large search in the matrix. 
        Args: 
            - Img: Input Image.
        Returns: 
            - Dg: Descritor array. 
    """

    # gets the size of the input matrix (M x N)
    M,N = Img.shape #all the images are the same size 
    
    # Defining the sobel operator 
    wsx =  np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) 
    wsy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Defining the size of the gradients gx and gy 
    gx = np.zeros((M, N)) 
    gy = np.zeros((M, N))

    # Calculating the convolution from each image with the operator 
    gx = ndimage.convolve(Img, wsx)
    gy = ndimage.convolve(Img, wsy)

    # Calculating the sum_jj
    sum_jj = np.sum(np.sqrt(gx ** 2 + gy ** 2))

    # Calculating the Magnitude Matrix
    Mxy = np.sqrt(gx ** 2 + gy ** 2) / sum_jj

    # Calculating the Angles matrix 
    Phixy = np.zeros((M, N)) 
    for i in range(M):
        for j in range(N):
            Phixy[i][j]= np.arctan(gy[i][j]/gx[i][j])
            
    # Sum pi/2 to all the elements in the Phixy vector
    Phixy += math.pi / 2

    # Converting the angles from rads to degree
    Phixy = np.rad2deg(Phixy)
     
    # for each Phixy(i, j), determine in which of the 9 bins the value falls into.
    # Phyxyd stores the the bins for each position 
    Phyxyd = Phixy//20

    #defining the descritor array  
    Dg = np.zeros(9) 

    # Calculating the descriptor array 
    for k in range(9): #size of the Descritor array (Dg) that matchs with the bins 
        for i in range(M):
            for j in range(N):
                if Phyxyd[i][j] == k:
                    Dg[k] =  Dg[k] + Mxy[i][j]
    
    return(Dg)
def KNN(Dg_0, Dg_1, Dg_test, K):
    """
        KNN function performs k-nearest neighbors classification on a given test image.
    
        Args: 
            - Dg_0: List of HoG descriptors for the X0 collection. 
            - Dg_1: List of HoG descriptors for the X1 collection.
            - Dg_test: HoG descriptor for the test image. 
            - K : Number of training images that can get “a vote”.

        Returns:
            - int: The predicted class (0 or 1) for the test image.

    """

    # Defining a distance list for the test image from the training images. 
    dist =[]
    Dg = Dg_0 + Dg_1
    for i in range(len(Dg)): # Loop for the number of images in the collection
        dist_aux = np.sqrt(np.sum((Dg[i] - Dg_test)**2 ))
        dist.append(dist_aux)
    
    
    Ksmall = heapq.nsmallest(K, dist)
    Ksmall_index = [] 

    for i in range(K):
        index_aux = dist.index(Ksmall[i])
        Ksmall_index.append(index_aux)
    
    votes_for_0 = np.sum(np.array(Ksmall_index) < len(Dg_0))
    votes_for_1 = K - votes_for_0

    if votes_for_0 > votes_for_1:
        return(0)
    else:
        return(1)
    
#--------------------------------------------------------------------------------------------------------------------------
#Importing the necessary libraries. 
import numpy as np 
from scipy import ndimage
import heapq
import math
import imageio.v2 as imageio
import matplotlib.pyplot as plt

#STEP 1: Load the three image collections X0 = {p0, · · · , pn} , X1 = {p0, · · · , pm} and Xtest = {t0, · · · , to}, 
# containing images without humans (X0), images with humans (X1) and test images (Xtest). 

# Reading the collections X0, X1 and Xtest
filenameX0 = input().rstrip()
X0 = read_image_collection(filenameX0)

filenameX1 = input().rstrip()
X1 = read_image_collection(filenameX1)

filenameXtest = input().rstrip()
Xtest = read_image_collection(filenameXtest)

#STEP 2: Transform all images to black&white using the Luminance technique
#STEP 2: Transform all images to black&white using the Luminance technique
X0_bw = [Luminance(img) for img in X0]
X1_bw = [Luminance(img) for img in X1]
Xtest_bw = [Luminance(img) for img in Xtest]

#STEP 3: Implement and compute the HoG descriptor of all images. 

# For the images in the X0 collection (X0_bw)
Dg_0 = [] 
for i in range(len(X0_bw)):
    dg_aux = HoG(X0_bw[i])
    Dg_0.append(dg_aux)

# For the images in the X1 collection (X1_bw)
Dg_1 = [] 
for i in range(len(X1_bw)):
    dg_aux = HoG(X1_bw[i])
    Dg_1.append(dg_aux)

Dg_test = [] 
for i in range(len(Xtest_bw)):
    dg_aux = HoG(Xtest_bw[i])
    Dg_test.append(dg_aux)

#STEP 4: Implement and Train a K-nearest-neighbours model with K = 3 using the training images X0 and X1. 

#passing the teste images to the KNN function one by one 

test_class = [] # list of values (0 or 1) that stores the class of each test image 
for i in range(len(Xtest)):
    class_aux = KNN(Dg_0, Dg_1, Dg_test[i], 3)
    test_class.append(class_aux)

print(*test_class, sep = " ")