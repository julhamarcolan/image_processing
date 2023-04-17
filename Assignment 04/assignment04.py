"""Assignment 04: Mathematical morphology
    Name: Julia Marcolan Teixeira 
    Usp Number: 10085370 
    Course code: MAI5021
    Year/semester: 2023.1

Task: The objective of this task is to create and execute the Flood Fill Algorithm for the purpose of painting a specific region
and detecting connected components within an image. The algorithm must be capable of producing the pixels belonging to the region’s 
connected components as output. 
"""
#--------------------------------------------------------------------------------------------------------------------------
#Importing the necessary libraries. 
import numpy as np 
import imageio.v2 as imageio
import matplotlib.pyplot as plt

#STEP 1: Read parameters (Binary image, Seed pixel x coordinate, Seed pixel y coordinate, Connectivity) and load binary image

#Reading the binary image 
filename =  input().rstrip()
b_img = (imageio.imread(filename) > 127).astype(np.uint8)

#Reading Seed pixel x coordinate
x = input().rstrip() 

#Reading Seed pixel y coordinate
y = input().rstrip() 

#Reading Connectivity
c = input().rstrip() 



