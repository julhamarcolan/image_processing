"""Assignment 01: enhancement and superresolution
    Name: Julia Marcolan Teixeira 
    Course code: MAI5020
    Year/semester: 2023.1

Task: Implement 3 distinct image enhancement techniques, as well as a superresolution method based on multiple views of the 
same image. 
"""
def Histogram(Img, n_glevels):
    """ 
        Histogram function returns the image  histogram used in the enhancement method  (obs: just the simple histogram, not cumulative)
        
        Args: 
            - Img: Input image.
            - n_glevels: Number og grey levels.
        
        Returns:
            - hist: The image  histogram. 
    """
    # creates an empty histogram 
    hist = np.zeros(n_glevels).astype(int)

    for i in range(n_glevels):
        # The np.sum, returns a sum of array elements over a given axis. In this case, it sum all positions in which
        # Img == i is true (i.e. all the pixels in the image where the glevel is i)
        pixels_value_i = np.sum(Img == i)
        # store it in the histogram array
        hist[i] = pixels_value_i

    return(hist)

def Joint_Histogram(Img, n_glevels, N_images):
    """
        The function Joint_Histogram is used to calculate the Joint Cumulative Histogram. The function calls the Histogram to 
        calculate the indivial Histogram for each image, sum them all and after that calculates te cumulative Histogram. 

        Args: 
            - Img: List of input images. 
            - n_glevels: the number of grey levels.
            - N_images: Number of images.
        Returns: 
            JHist_c: Joint Cumulative Histogram.
    """
    hist_i = [] #List containg all the histograms of each imaga hist_i[0] -> histogram from the imag0
    hist_aux =  np.zeros(n_glevels).astype(int) #Auxiliar vector used to store the histogram from each image 
    for i in range(N_images):
       hist_aux =  Histogram(Img[i], n_glevels)
       hist_i.append(hist_aux)

        # sum the individual histograms to get the joint histogram
    join_hist = np.sum(hist_i, axis=0)

    # compute the joint cumulative histogram
    join_hist_c = np.cumsum(join_hist)

    return join_hist_c
            
def equalize_histogram(Img, n_glevels, sh = None, N_images = None):
    """
    The function equalize_histogram utilizes the Histogram Equalization method, which employs
    a cumulative histogram and a transformation function to equalize the histogram of each image. 
        
    For the Single-image Cumulative Histogram, the transformation function is given by: 
    s = T(z) = ((L - 1)/ (M  * N) ) * Hist_c,
    where (L - 1) represents the maximum grey level value, M and N are the matrix/image dimensions, and Hist_c is the cumulative histogram.

    For the Joint Cumulative Histogram, it's necessary to normalize the histogram considering the number of images. In this case,
    the transformation function is given by: 
    s = T(z) = ((L - 1) / (N_images * M * N)) * Hist_c,
    where N_images is the number of images.

    Args:
        - Img: Input image.
        - n_glevels: the number of grey levels.
        - sh: if a value for sh is passed to the function, the Joint Cumulative Histogram will be used. 
        By default, this parameter is set to None, indicating the use of the Single-image Cumulative Histogram.
        - N_images: the number of images used in the Joint Cumulative Histogram. If the Single-image Cumulative Histogram is used, 
        this parameter is ignored.
    
    Returns:
        - Img_eq: The outpu image/matrix  transformed/equalized.
        - transform_func: The transformation function.
    """

    # creates an empty vector to store the  transformation function
    transform_func = np.zeros(n_glevels).astype(np.uint8)

    # gets the size of the input matrix (M x N)
    M, N = Img.shape

    # create the image to store the equalised version
    Img_eq = np.zeros([M,N]).astype(np.uint8)

    # obtaining the cumulative histogram. If the sh argument is none, then a cumulative histogram is obtained for the single image, 
    # else the function uses the single histogram (sh) calculated by the def function  Joint_Histogram for a range of images. sh is passed 
    # as a function argument. 
    if sh is None: 
        hist = Histogram(Img, n_glevels)
        # creates an empty cumulative  histogram hist_c 
        hist_c = np.zeros(n_glevels).astype(int)
        # computes the cumulative histogram hist_c 
        hist_c[0] = hist[0] 
        for i in range(1,  n_glevels):
            hist_c[i] = hist[i] + hist_c[i-1]

        # Each intensity value, is transform into a new intensity
        for z in range(n_glevels):
            #s = T(z) = ((L - 1)/ (M * N) ) * Hist_c
            s = ((n_glevels-1)/float(M*N))*hist_c[z]

            # for every coordinate in which matrix Img has the value 'z'
            # assigns the transformed value 's'
            Img_eq[ np.where(Img == z) ] = s
        
            # store the tranformation function
            transform_func[z] = s
            
        return(Img_eq, transform_func)
    else: 
        hist_c = sh
        # Each intensity value, is transform into a new intensity
        for z in range(n_glevels):
            #s = T(z) = ((L - 1)/ (M * N * N_images) ) * Hist_c
            s = ((n_glevels-1)/float(M*N*N_images))*hist_c[z]

            # for every coordinate in which matrix Img has the value 'z'
            # assigns the transformed value 's'
            Img_eq[ np.where(Img == z) ] = s
        
            # store the tranformation function
            transform_func[z] = s
            
        return(Img_eq, transform_func)

def Gamma_Correction(Img, n_glevels, gamma): 
    """
        The Gamma correction function  transform eavh input pixels z from the Img that are within some intensity level interval,
        into an output pixel s from the img_gamma. The Gamma Transformation is given by: 

        s = (L - 1) ( z / (L-1))^1/gamma, where 
        z is the input image, s is the output image and (L - 1) represents the maximum grey level in the input image.  

        Args: 
            - Img: Input image. 
            - n_glevels: Number of grey levels.
        Returns: 
            img_gamma: output image corretec by the gamma correction.  
    """

    # gets the size of the input matrix (M x N)
    M, N = Img.shape

    #Calculating the inv Gamma 
    invGamma = 1.0/gamma

    # create the image to store the corrected version
    img_gamma = np.zeros([M,N]).astype(np.uint8)

    #Gamma Correction  s = (L - 1) ( z / (L-1))^1/gamma
    img_gamma = ((n_glevels - 1) * np.power(Img.astype(np.int32)/ (n_glevels - 1), invGamma)).astype(np.uint8) 
    return(img_gamma)

def Superresolution(Img): 
    """
        The function calculates a High resolution funcion by adding 4 low resolution functions ( l1 = Img[0], l2 = Img[1], l3 = Img[2],
        l4 = Img[3]). 

        Args: 
            - Img: list of low resolution input images. 
        Returns:
            - H: High reolution output image created uding the superresolution method. 
    """

    # gets the size of the input matrix (M x N). Obs: The four input images have the same shape and are stored in the 
    # Img list
    M, N = Img[0].shape

    #Creating the matrix H with the double of the size 
    H = np.zeros([2*M,2*N])

    #Calculating the H components,extract from l1 = Img[0]
    for i in range(M):
        for j in range(N):
            I = 2*i 
            J = 2*j
            H[I][J] = Img[0][i][j]

    #Calculating the H components,extract from l2 = Img[1]
    for i in range(M):
        for j in range(N):
            I = 2*i 
            J = (2*j) + 1 
            H[I][J] = Img[1][i][j]

    #Calculating the H components,extract from l3 = Img[2]
    for i in range(M):
        for j in range(N):
            I = (2*i) + 1  
            J = 2*j
            H[I][J] = Img[2][i][j]
    
    #Calculating the H components,extract from l4 = Img[3]
    for i in range(M):
        for j in range(N):
            I = (2*i) + 1  
            J = (2*j) + 1 
            H[I][J] = Img[3][i][j]

    return (H)

def RMSE(H, imgH): 
    """
        The function calculates the root mean square (RMSE) value between the image obteined by the Superresolution image 
        and a reference High resolution image. 
        
        Args: 
            - H: Input image 1 (image obteined by using the Superresolution Method).
            - imgH: High resolution reference image. 
    """

     # gets the size of the input matrix (M x N). Obs: H and imgH should have the same size 
    M, N = H.shape

    sum_squares = np.sum(np.square(imgH - H))

    RMSE_value = np.sqrt(sum_squares/(M*N))

    return(RMSE_value)

#--------------------------------------------------------------------------------------------------------------------------
#Importing the necessary libraries. 
import numpy as np 
import imageio.v2 as imageio
import matplotlib.pyplot as plt

#STEP 1: Reading the input parameters: The low resolution images, the high resolution, the enhancement method identifier F,
# and the enhancement method parameter gamma

#Oppening the low resolution images 
img = [] #list of the images img[0] refers to the 01_low1.png 01_high.png
filename =  input().rstrip()
for i in range(4):
    filename_aux = f"{filename}{i}.png"
    img_aux = imageio.imread(filename_aux)
    img.append(img_aux)

# Oppening the high resolution reference image (imgH)
filenameH =  input().rstrip()
imgH = imageio.imread(filenameH)

# Selecting the enhancement method identifier:
# 0: No Enhancement;  1: Single-image Cumulative Histogram; 2: Joint Cumulative Histogram; 3: Gamma Correction Function
F_value_aux =  input().rstrip()
F_value = int(F_value_aux)

# Gamma parameter: If F_value was differente from 3, the gamma parameter is ignorated
gamma_aux =  input().rstrip()
gamma = float(gamma_aux)

#STEP 2: Selecting the correct image enhancement method 
if F_value == 0: 
    #No Enhancement
    img_process = img
elif F_value == 1:
    #Single-image Cumulative Histogram: Compute the Cumulative Histogram for each image and use it as a transform function
    #to equalize the histogram of each image

    img_process = [] #list of equalized images 
    eq_transform = [] #list of transformation functions for each image 
    for i in range(4):
        img_eq_aux, eq_transform_aux = equalize_histogram(img[i], 256) #8bit image, sh = None, N_images = None
        img_process.append(img_eq_aux)
        eq_transform.append(eq_transform_aux)

elif F_value == 2:
    #Joint Cumulative Histogram: Compute a single Cumulative Histogram over all images and use it as a transform function to equalize each imag
    sh = Joint_Histogram(img, 256, 4)
    img_process = [] #list of equalized images 
    eq_transform = [] #list of transformation functions for each image 
    for i in range(4):
        img_eq_aux, eq_transform_aux = equalize_histogram(img[i], 256, sh, 4) 
        img_process.append(img_eq_aux)
        eq_transform.append(eq_transform_aux)

elif F_value == 3:
    #Gamma Correction Function: Implement the pixel-wise enhancement function called Gamma Correction
    img_process = [] #list of images corrected by the gamma conrrection 
    for i in range(4):
        img_gamma_aux = Gamma_Correction(img[i], 256, gamma)
        img_process.append(img_gamma_aux)

#STEP 3 : Superresolution
H = Superresolution(img_process)
RMSE_value = RMSE(H, imgH) 
RMSE_value = round(RMSE_value, 4)
print(RMSE_value)
