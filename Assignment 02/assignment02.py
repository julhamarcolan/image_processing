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

def Ideal_Low_pass(f, r):
    """
        Apply an ideal low-pass filter to the input image in the frequency domain. The ideal has circular shape with radios r. 

        STEPS: 
        1. The function computes the 2D Fourier transform of the input image f using np.fft.fft2 and shifts the result to center 
        the low frequencies using np.fft.fftshift. 
        2. The function creates a circular ideal low-pass filter in the frequency domain, with radius r:
        Given the function  D = np.sqrt((u- M/2)**2 + (v-N/2)**2), if D <= r -> H = 1 else H = 0. 
        3. The filtered image is obtained  by element-wise multiplication of the Fourier transform of the input image and the ideal low-pass filter. 
        4. The inverse Fourier  transform is calculated using np.fft.ifft2 after shifting the filtered image back to the original position using 
        np.fft.ifftshift.
        5. The resulting filtered image g is then rescaled to the range [0-255] using min-max normalization, and converted to int32 data type. 
        6. Besides the reconstructed image, the function also returns que magnitude of the Fourier Transform and the low-pass filter 
        created by the function  in order to testing. 

        Args: 
            - f: Input image to be filtered.
            - r: Radius of the circular filter in the frequency domain.
      
        Returns: 
            - np.abs(F): 2D array representing the magnitude of the  Fourier transform of the input image.
            - H: 2D array representing the ideal low-pass filter in the frequency domain.
            - g: Restored image. 
    """
    # Computing the 2D Fourier transform of the input image f
    F = np.fft.fftshift(np.fft.fft2(f))

    # Creating the low-pass filter 
    M, N = F.shape
    H = np.zeros((M, N), dtype=np.float32)

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u- M/2)**2 + (v-N/2)**2)
            if D <= r:
                H[u,v] = 1
            else:
                H[u,v] = 0
    
    # Filtering the image 
    G = F * H 

    # Calculating the inverse Fourier Transform 
    g_shift  = np.fft.ifftshift(G)
    g = np.real(np.fft.ifft2(g_shift))

    # Rescaling the image to the range [0-255]
    min_value = np.min(g)
    max_value = np.max(g)
    g = (g - min_value) * 255 / (max_value - min_value)

    #Converting the image to 8-bit  integer (uint8)
    g = g.astype(np.int32)

    return(np.abs(F), H, g)

def Ideal_High_pass(f, r):
    """
        Apply an ideal high-pass filter to the input image in the frequency domain. The ideal has circular shape with radios r. 

        STEPS: 
        1. The function computes the 2D Fourier transform of the input image f using np.fft.fft2 and shifts the result to center 
        the low frequencies using np.fft.fftshift. 
        2. The function creates a circular ideal high-pass filter in the frequency domain, with radius r:
        Given the function  D = np.sqrt((u- M/2)**2 + (v-N/2)**2), if D <= r -> H = 0 else H = 1.
        3. The filtered image is obtained  by element-wise multiplication of the Fourier transform of the input image and the ideal high-pass filter. 
        4. The inverse Fourier  transform is calculated using np.fft.ifft2 after shifting the filtered image back to the original position using 
        np.fft.ifftshift.
        5. The resulting filtered image g is then rescaled to the range [0-255] using min-max normalization, and converted to int32 data type. 
        6. Besides the reconstructed image, the function also returns que magnitude of the Fourier Transform and the high-pass filter 
        created by the function  in order to testing. 

        Args: 
            - f: Input image to be filtered.
            - r: Radius of the circular filter in the frequency domain.
      
        Returns: 
            - np.abs(F): 2D array representing the magnitude of the  Fourier transform of the input image.
            - H: 2D array representing the ideal high-pass filter in the frequency domain.
            - g: Restored image. 
    """
    # Computing the 2D Fourier transform of the input image f
    F = np.fft.fftshift(np.fft.fft2(f))

    # Creating the high-pass filter (H_high_pass = 1 - H_low_pass)
    M, N = F.shape
    H = np.zeros((M, N), dtype=np.float32)

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u- M/2)**2 + (v-N/2)**2)
            if D <= r:
                H[u,v] = 0
            else:
                H[u,v] = 1
    
    # Filtering the image 
    G = F * H 

    # Calculating the inverse Fourier Transform 
    g_shift  = np.fft.ifftshift(G)
    g = np.real(np.fft.ifft2(g_shift))

    # Rescaling the image to the range [0-255]
    min_value = np.min(g)
    max_value = np.max(g)
    g = (g - min_value) * 255 / (max_value - min_value)

    #Converting the image to int32
    g = g.astype(np.int32)

    return(np.abs(F), H, g) 

def Ideal_Band_pass(f, r1, r2):
    """
        The Ideal_Band_pass function implement a band-pass filter on an input image .
        
        STEPS: 
        1. The function computes the 2D Fourier transform of the input image f using np.fft.fft2 and shifts the result to center 
        the low frequencies using np.fft.fftshift.
        2. Set the values in H to 1 if D is within the range [r1, r2], and 0 otherwise, effectively creating a band-pass filter.
        3. The filtered image is obtained  by element-wise multiplication of the Fourier transform of the input image and the ideal 
        band-pass filter. 
        4. The inverse Fourier  transform is calculated using np.fft.ifft2 after shifting the filtered image back to the original position using 
        np.fft.ifftshift.
        5. The resulting filtered image g is then rescaled to the range [0-255] using min-max normalization, and converted to int32 data type. 
        6. Besides the reconstructed image, the function also returns que magnitude of the Fourier Transform and the high-pass filter 
        created by the function  in order to testing.

        Args: 
            - f: Input image to be filtered.
            - r1: Outer  radius of the band-pass filter.
            - r2: Inner radius of the band-pass filter.

        Returns: 
            - np.abs(F): 2D array representing the magnitude of the  Fourier transform of the input image.
            - H: 2D array representing the  band-pass filter in the frequency domain.
            - g: Restored image. 
    """
    # Computing the 2D Fourier transform of the input image f
    F = np.fft.fftshift(np.fft.fft2(f))

    # Creating the Band-pass filter 
    M, N = F.shape
    H = np.zeros((M, N), dtype=np.float32)

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u- M/2)**2 + (v-N/2)**2)
            if D <= r1 and D >= r2:
                H[u,v] = 1
            else:
                H[u,v] = 0
        

    # Filtering the image 
    G = F * H 

    # Calculating the inverse Fourier Transform 
    g_shift  = np.fft.ifftshift(G)
    g = np.real(np.fft.ifft2(g_shift))

    # Rescaling the image to the range [0-255]
    min_value = np.min(g)
    max_value = np.max(g)
    g = (g - min_value) * 255 / (max_value - min_value)

    #Converting the image to 8-bit  integer (uint8)
    g = g.astype(np.int32)

    return(np.abs(F), H, g) 

def Laplacian_High_pass(f):
    """
        Apply  a Laplacian high pass filter in the input image. 

        STEPS: 
        1. The function computes the 2D Fourier transform of the input image f using np.fft.fft2 and shifts the result to center 
        the low frequencies using np.fft.fftshift.
        2. The function creates a Laplacian high pass filter in the frequency domain:
        H[u,v] = -4*(np.pi)**2*((u-M/2)**2 + (v-N/2)**2), and the filter is given by  H = 255 - H 
        3. The filtered image is obtained  by element-wise multiplication of the Fourier transform of the input image and the ideal 
        band-pass filter. 
        4. The inverse Fourier  transform is calculated using np.fft.ifft2 after shifting the filtered image back to the original position using 
        np.fft.ifftshift.
        5. The resulting filtered image g is then rescaled to the range [0-255] using min-max normalization, and converted to int32 data type. 
        6. Besides the reconstructed image, the function also returns que magnitude of the Fourier Transform and the high-pass filter 
        created by the function  in order to testing.

        Args: 
            - f: Input image to be filtered.

        Returns: 
            - np.abs(F): 2D array representing the magnitude of the  Fourier transform of the input image.
            - H: 2D array representing the Laplacian high pass filter in the frequency domain.
            - g: Restored image. 
    """
    # Computing the 2D Fourier transform of the input image f
    F = np.fft.fftshift(np.fft.fft2(f))

    # Creating the Laplacian low-pass filter  
    M, N = F.shape
    H = np.zeros((M, N), dtype=np.float32)

    for u in range(M):
        for v in range(N):
            H[u,v] = -4*np.pi**2*((u-M/2)**2 + (v-N/2)**2)
    H = 255 - H 
    
    # Filtering the image 
    G = F * H 

    # Calculating the inverse Fourier Transform 
    g_shift  = np.fft.ifftshift(G)
    g = np.real(np.fft.ifft2(g_shift))

    # Rescaling the image to the range [0-255]
    min_value = np.min(g)
    max_value = np.max(g)
    g = (g - min_value) * 255 / (max_value - min_value)

    #Converting the image to 8-bit  integer (uint8)
    g = g.astype(np.int32)

    return(np.abs(F), H, g) 

def Gaussian_Low_pass(f, sigma1, sigma2):
    """
        Apply  a Gaussian low pass filter in the input image. 

        STEPS: 
        1. The function computes the 2D Fourier transform of the input image f using np.fft.fft2 and shifts the result to center 
        the low frequencies using np.fft.fftshift.
        2. The function creates a Gaussian low pass filter in the frequency domain
        H[u,v] = np.exp(-x) where x = ((u - M/2)**2) / (2*sigma1**2) + ((v - N/2)**2) / (2*sigma2**2)
        3. The filtered image is obtained  by element-wise multiplication of the Fourier transform of the input image and the ideal 
        band-pass filter. 
        4. The inverse Fourier  transform is calculated using np.fft.ifft2 after shifting the filtered image back to the original position using 
        np.fft.ifftshift.
        5. The resulting filtered image g is then rescaled to the range [0-255] using min-max normalization, and converted to int32 data type. 
        6. Besides the reconstructed image, the function also returns que magnitude of the Fourier Transform and the high-pass filter 
        created by the function  in order to testing.

        Args: 
            - f: Input image to be filtered.
            - sigma1: standard deviation. 
            - sigma2: standard deviation. 

        Returns: 
            - np.abs(F): 2D array representing the magnitude of the  Fourier transform of the input image.
            - H: 2D array representing the Gaussian low pass filter in the frequency domain.
            - g: Restored image. 
    """
    # Computing the 2D Fourier transform of the input image f
    F = np.fft.fftshift(np.fft.fft2(f))

     # Creating the Laplacian low-pass filter  
    M, N = F.shape
    H = np.zeros((M, N), dtype=np.float32)

    for u in range(M):
        for v in range(N):
            l1 = ((u - M/2)**2) / (2*sigma1**2)
            l2 = ((v - N/2)**2) / (2*sigma2**2)
            x = l1 + l2 
            H[u,v] = np.exp(-x)
    
    # Filtering the image 
    G = F * H 

    # Calculating the inverse Fourier Transform 
    g_shift  = np.fft.ifftshift(G)
    g = np.real(np.fft.ifft2(g_shift))

    # Rescaling the image to the range [0-255]
    min_value = np.min(g)
    max_value = np.max(g)
    g = (g - min_value) * 255 / (max_value - min_value)

    #Converting the image to 8-bit  integer (uint8)
    g = g.astype(np.int32)

    return(np.abs(F), H, g) 

def Butterworth_low_pass(f, N0, D0):
    """
        Apply  a  Butterworth low pass filter in the input image. 

        STEPS: 
        1. The function computes the 2D Fourier transform of the input image f using np.fft.fft2 and shifts the result to center 
        the low frequencies using np.fft.fftshift.
        2. The function creates a  filter in the frequency domain
        H[u,v] = 1 / (1 + (D/D0)**N0), where  D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        3. The filtered image is obtained  by element-wise multiplication of the Fourier transform of the input image and the ideal 
        band-pass filter. 
        4. The inverse Fourier  transform is calculated using np.fft.ifft2 after shifting the filtered image back to the original position using 
        np.fft.ifftshift.
        5. The resulting filtered image g is then rescaled to the range [0-255] using min-max normalization, and converted to int32 data type. 
        6. Besides the reconstructed image, the function also returns que magnitude of the Fourier Transform and the high-pass filter 
        created by the function  in order to testing.

        Args: 
            - f: Input image to be filtered.
            - N0: order from the transfer function of the Butterworth lowpass filter. 
            - D0: distance from the origin.

        Returns: 
            - np.abs(F): 2D array representing the magnitude of the  Fourier transform of the input image.
            - H: 2D array representing the  Butterworth low pass filter in the frequency domain.
            - g: Restored image. 
    """

    # Computing the 2D Fourier transform of the input image f
    F = np.fft.fftshift(np.fft.fft2(f))

     # Creating the Laplacian low-pass filter  
    M, N = F.shape
    H = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = 1 / (1 + (D/D0)**N0)
    # Filtering the image 
    G = F * H 

    # Calculating the inverse Fourier Transform 
    g_shift  = np.fft.ifftshift(G)
    g = np.real(np.fft.ifft2(g_shift))

    # Rescaling the image to the range [0-255]
    min_value = np.min(g)
    max_value = np.max(g)
    g = (g - min_value) * 255 / (max_value - min_value)

    #Converting the image to 8-bit  integer (uint8)
    g = g.astype(np.int32)

    return(np.abs(F), H, g) 

def Butterworth_high_pass(f, N0, D0):
    """
    
    """

    # Computing the 2D Fourier transform of the input image f
    F = np.fft.fftshift(np.fft.fft2(f))

     # Creating the Laplacian low-pass filter  
    M, N = F.shape
    H = np.zeros((M, N), dtype=np.float32)
    
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = 1 / (1 + (D0/D)**N0)
    # Filtering the image 
    G = F * H 

    # Calculating the inverse Fourier Transform 
    g_shift  = np.fft.ifftshift(G)
    g = np.real(np.fft.ifft2(g_shift))

    # Rescaling the image to the range [0-255]
    min_value = np.min(g)
    max_value = np.max(g)
    g = (g - min_value) * 255 / (max_value - min_value)

    #Converting the image to 8-bit  integer (uint8)
    g = g.astype(np.int32)

    return(np.abs(F), H, g) 

def RMSE(img_H, img_R): 
    """
        The function calculates the root mean square (RMSE) value between the restored image (img_R) obteined after the filter application 
        and a reference/expected image (img_H)
        
        Args: 
            - img_H: Expected image. 
            - img_R: Restored image.
        Return: 
            - RMSE_value: root mean square (RMSE) value between the images.
    """

    # gets the size of the input matrix (M x N). Obs: H and img_R should have the same size 
    M, N = img_H.shape

    # Converting the image to type np.int32 to avoid overflow 
    img_H = img_H.astype(np.int32)
    img_R = img_R.astype(np.int32)

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

    #Applying the Low-pass filter 
    F, H, g = Ideal_Low_pass(img_I, r)
    RMSE_value = RMSE(img_H, g)
    RMSE_value = round(RMSE_value, 4)
    print(RMSE_value)

    #Just  for testing
    #plt.figure(figsize=(12,10))
    #plt.subplot(221); plt.imshow(img_I, cmap ='gray')
    #plt.subplot(222); plt.imshow(F, cmap ='gray')
    #plt.subplot(223); plt.imshow(H, cmap ='gray')
    #plt.subplot(224); plt.imshow(g, cmap ='gray')
    #plt.show()

elif index == 1: 
    # Ideal High-pass
    r = int(input().rstrip())

    #Applying the  High-pass  filter 
    F, H, g = Ideal_High_pass(img_I, r)
    RMSE_value = RMSE(img_H, g)
    RMSE_value = round(RMSE_value, 4)
    print(RMSE_value)

    #Just  for testing
    plt.figure(figsize=(12,10))
    plt.subplot(221); plt.imshow(img_I, cmap ='gray')
    plt.subplot(222); plt.imshow(F, cmap ='gray')
    plt.subplot(223); plt.imshow(H, cmap ='gray')
    plt.subplot(224); plt.imshow(g, cmap ='gray')
    plt.show()

elif index == 2: 
    # Ideal Band-pass
    r1 = int(input().rstrip())
    r2 = int(input().rstrip())

    #Applying the  Band-pass  filter 
    F, H, g = Ideal_Band_pass(img_I, r1, r2)
    RMSE_value = RMSE(img_H, g)
    RMSE_value = round(RMSE_value, 4)
    print(RMSE_value)

    #Just  for testing
    plt.figure(figsize=(12,10))
    plt.subplot(221); plt.imshow(img_I, cmap ='gray')
    plt.subplot(222); plt.imshow(F, cmap ='gray')
    plt.subplot(223); plt.imshow(H, cmap ='gray')
    plt.subplot(224); plt.imshow(g, cmap ='gray')
    plt.show()

elif index == 3 : 
    #Laplacian low-pass

    #Applying the Laplacian low-pass filter 
    F, H, g = Laplacian_High_pass(img_I)
    RMSE_value = RMSE(img_H, g)
    RMSE_value = round(RMSE_value, 4)
    print(RMSE_value)

    plt.figure(figsize=(12,10))
    plt.subplot(221); plt.imshow(img_I, cmap ='gray')
    plt.subplot(222); plt.imshow(F, cmap ='gray')
    plt.subplot(223); plt.imshow(H, cmap ='gray')
    plt.subplot(224); plt.imshow(g, cmap ='gray')
    plt.show()

elif index == 4: 
    # Gaussian Low-pass
    sigma1 = int(input().rstrip())
    sigma2 = int(input().rstrip())
    
    #Applying the Laplacian low-pass filter 
    F, H, g = Gaussian_Low_pass(img_I, sigma1, sigma2)
    RMSE_value = RMSE(img_H, g)
    RMSE_value = round(RMSE_value, 4)
    print(RMSE_value)

    plt.figure(figsize=(12,10))
    plt.subplot(221); plt.imshow(img_I, cmap ='gray')
    plt.subplot(222); plt.imshow(F, cmap ='gray')
    plt.subplot(223); plt.imshow(H, cmap ='gray')
    plt.subplot(224); plt.imshow(g, cmap ='gray')
    plt.show()

elif index == 5: 
    # Butterworth low-pass
    D0 = int(input().rstrip())
    N = int(input().rstrip())

    #Applying the  High-pass  filter 
    F, H, g = Butterworth_low_pass(img_I, N, D0)
    RMSE_value = RMSE(img_H, g)
    RMSE_value = round(RMSE_value, 4)
    print(RMSE_value)

    #Just  for testing
    plt.figure(figsize=(12,10))
    plt.subplot(221); plt.imshow(img_I, cmap ='gray')
    plt.subplot(222); plt.imshow(F, cmap ='gray')
    plt.subplot(223); plt.imshow(H, cmap ='gray')
    plt.subplot(224); plt.imshow(g, cmap ='gray')
    plt.show()
elif index == 6: 
    # Butterworth high-pass
    D0 = int(input().rstrip())
    N = int(input().rstrip())

    #Applying the  High-pass  filter 
    F, H, g = Butterworth_high_pass(img_I, N, D0)
    RMSE_value = RMSE(img_H, g)
    RMSE_value = round(RMSE_value, 4)
    print(RMSE_value)

    #Just  for testing
    plt.figure(figsize=(12,10))
    plt.subplot(221); plt.imshow(img_I, cmap ='gray')
    plt.subplot(222); plt.imshow(F, cmap ='gray')
    plt.subplot(223); plt.imshow(H, cmap ='gray')
    plt.subplot(224); plt.imshow(g, cmap ='gray')
    plt.show()
else: 
    print("Index not valid")
