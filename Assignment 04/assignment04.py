"""Assignment 04: Mathematical morphology
    Name: Julia Marcolan Teixeira 
    Usp Number: 10085370 
    Course code: MAI5021
    Year/semester: 2023.1

Task: The objective of this task is to create and execute the Flood Fill Algorithm for the purpose of painting a specific region
and detecting connected components within an image. The algorithm must be capable of producing the pixels belonging to the region’s 
connected components as output. 
"""
#Importing the necessary libraries. 
import numpy as np 
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def flood_fill_4n(x, y, img, target_color, coords_list): 
    """
        Recursive implementation of 4-connected neighborhood flood fill algorithm. The algorithm  searches all nodes 
        in the matrix that are connected to the initial point by a path of the target color and modifies them to the 
        replacement color. The function has no returns and the args img and coords_list are passed as reference. 
        Args: 
            - x: x coordinate from the seed pixel. 
            - y: y  coordinate from the seed pixel. 
            - img: Binary image to be painted. 
            - target_color: initial color from the seed pixel.
            - coords_list: List that stores the pixels where the color was change. 
        Returns: 
            ...  
    """
    M,N = img.shape
    
    # Selecting the target and the fill colors 
    fill_color = 1 - target_color
    
    if ( x < 0 or x > M - 1) or (y < 0 or y > N - 1):
        return

    # Changing (or not) the pixel color
    if img[x][y] != target_color: 
        return
    else: 
        img[x][y] = fill_color
        coords_list.append([x,y])

    # Recursive call from the 4-neighbors
    flood_fill_4n(x+1, y, img, target_color, coords_list)
    flood_fill_4n(x-1, y, img, target_color, coords_list)
    flood_fill_4n(x, y+1, img, target_color, coords_list)
    flood_fill_4n(x, y-1, img, target_color, coords_list)

def flood_fill_8n(x, y, img, target_color, coords_list): 
    """
        Recursive implementation of 8-connected neighborhood flood fill algorithm. The algorithm  searches all nodes 
        in the matrix that are connected to the initial point by a path of the target color and modifies them to the 
        replacement color. The function has no returns and the args img and coords_list are passed as reference. 
        Args: 
            - x: x coordinate from the seed pixel. 
            - y: y  coordinate from the seed pixel. 
            - img: Binary image to be painted. 
            - target_color: initial color from the seed pixel.
            - coords_list: List that stores the pixels where the color was change. 
        Returns: 
            ...  
    """
    M,N = img.shape
    
    # Selecting the target and the fill colors 
    fill_color = 1 - target_color
    
    if ( x < 0 or x > M - 1) or (y < 0 or y > N - 1):
        return

    # Changing (or not) the pixel color
    if img[x][y] != target_color: 
        return
    else: 
        img[x][y] = fill_color
        coords_list.append([x,y])

    # Recursive call from the 8-neighbors
    flood_fill_8n(x+1, y, img, target_color, coords_list)
    flood_fill_8n(x-1, y, img, target_color, coords_list)
    flood_fill_8n(x, y+1, img, target_color, coords_list)
    flood_fill_8n(x, y-1, img, target_color, coords_list)
    flood_fill_8n(x+1, y+1, img, target_color, coords_list)
    flood_fill_8n(x+1, y-1, img, target_color, coords_list)
    flood_fill_8n(x-1, y-1, img, target_color, coords_list)
    flood_fill_8n(x+1, y+1, img, target_color, coords_list)

#--------------------------------------------------------------------------------------------------------------------------

#STEP 1: Read parameters (Binary image, Seed pixel x coordinate, Seed pixel y coordinate, Connectivity) and load binary image

#Reading the binary image 
filename =  input().rstrip()
img = (imageio.imread(filename) > 127).astype(np.uint8)

# Forcing the image in an np.array to make it easier to pass the variable as a reference to the function 
img = np.array(img) 

#making a copy just for testing and comparing
img_save = (imageio.imread(filename) > 127).astype(np.uint8)

#Reading Seed pixel x coordinate
x = int(input().rstrip()) 

#Reading Seed pixel y coordinate
y = int(input().rstrip()) 

#Reading Connectivity
c = int(input().rstrip()) 

# Creating a list to store the coordinates of the modified pixels
coords_list = [] 

# Selecting the correct connectivity value
if c == 4: 
    # Setting the correct start color in the seed pixel 
    target_color = img[x][y]
    flood_fill_4n(x, y, img, target_color, coords_list)
 
elif c == 8: 
    #Setting the correct start color in the seed pixel 
    target_color = img[x][y]
    flood_fill_8n(x, y, img, target_color, coords_list)
else: 
    print("Invilid Connectivity value ")


# Printing the coordinates of the modified pixels
coords_list.sort()

str_list = []
for i in range(len(coords_list)): 
    str_aux  = f"({coords_list[i][0]} {coords_list[i][1]})"
    str_list.append(str_aux)

print(*str_list, sep = " ")

#Just for testing 
#plt.figure(figsize=(12,10))
#plt.subplot(121); plt.imshow(img_save, cmap ='gray')
#plt.subplot(122); plt.imshow(img, cmap ='gray')
#plt.show()