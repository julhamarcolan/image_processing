"""Assignment 04: Mathematical morphology
    Name: Julia Marcolan Teixeira 
    Usp Number: 10085370 
    Course code: MAI5021
    Year/semester: 2023.1

Task: The objective of this task is to create and execute the Flood Fill Algorithm for the purpose of painting a specific region
and detecting connected components within an image. The algorithm must be capable of producing the pixels belonging to the region’s 
connected components as output. 
"""

def flood_fill_4n(x, y, img, target_color): 
    """
    
    """
    M,N = img.shape

    if target_color == 1: 
        fill_color = 0 
    else: 
        fill_color = 1
    
    if ( x < 0 or x > M - 1):
        return

    if (y < 0 or y > N - 1):
        return 

    if img[x][y] != target_color: 
        return
    else: 
        img[x][y] = fill_color

    flood_fill_4n(x+1, y, img, target_color)
    flood_fill_4n(x -1, y, img, target_color)
    flood_fill_4n(x, y+1, img, target_color)
    flood_fill_4n(x, y-1, img, target_color)
    

#--------------------------------------------------------------------------------------------------------------------------
#Importing the necessary libraries. 
import numpy as np 
import imageio.v2 as imageio
import matplotlib.pyplot as plt

#STEP 1: Read parameters (Binary image, Seed pixel x coordinate, Seed pixel y coordinate, Connectivity) and load binary image

#Reading the binary image 
#filename =  input().rstrip()
img = (imageio.imread('C:/Users/julia/Documents/ImageProcessing/Assignment 04/0.tiff') > 127).astype(np.uint8)
img = np.array(img) 
#img = np.array([[0,0,0], [1,0,1], [1,1,0]])

#Reading Seed pixel x coordinate
#x = int(input().rstrip()) 
x = 1
#Reading Seed pixel y coordinate
#y = int(input().rstrip()) 
y = 1
#Reading Connectivity
#c = input().rstrip() 
c = 4  
target_color = img[x][y]
flood_fill_4n(x,y, img, target_color)

#print(img)
#Just  for testing
plt.figure(figsize=(12,10))
plt.subplot(121); plt.imshow(img, cmap ='gray')
#plt.subplot(122); plt.imshow(img2, cmap ='gray')
plt.show()