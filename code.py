import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import matplotlib.image as mpimg

#### question (a)




#### question (b)

def GrayscaleChanger(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def double_threshold_hysteresis(img,lowThresholdRatio=0.05,highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res

def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])
            
            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed

def SobelFilter(image):
    image = GrayscaleChanger(GaussianBlur(image))
    convolved = np.zeros(image.shape)
    G_x = np.zeros(image.shape)
    G_y = np.zeros(image.shape)
    size = image.shape
    kernel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    kernel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            G_x[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_x))
            G_y[i, j] = np.sum(np.multiply(image[i - 1 : i + 2, j - 1 : j + 2], kernel_y))
    
    convolved = np.sqrt(np.square(G_x) + np.square(G_y))
    convolved = np.multiply(convolved, 255.0 / convolved.max())

    angles = np.rad2deg(np.arctan2(G_y, G_x))
    angles[angles < 0] += 180
    convolved = convolved.astype('uint8')
    return convolved, angles


def GaussianBlur(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image
def visualize(img, format=None, gray=False):
    plt.figure(figsize=(7,7))
    
    plt.imshow(img, format)
    


if __name__ == "__main__":
    print("select:\n1.question (a)\n2.question (b)")
    n=int(input("Enter the selected number '1'/'2': "))
    if(n==1):
    	
        kernel_size = 3
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        img = cv2.imread('Sample_flt.tif')
        plt.imshow(img,'gray')
        plt.title("full body")
        plt.show()
        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
        print("laplacian")
        plt.imshow(lap,'gray')
        plt.title("laplace")
        plt.show()



        img=np.uint8(np.absolute(img))
        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen_img_1=cv2.filter2D(img,-1,filter)
        img2 = np.uint8(np.absolute(lap))
        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen_img_2=cv2.filter2D(img2,-1,filter)
        sharpened = cv2.add(img,img2)
        print("sharpened")

        plt.imshow(sharpened,'gray',label="sharpened")
        plt.title('sharpened laplace image')
        plt.show()



        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)  
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1) 
        absx= np.uint8(np.absolute(sobelx))
        absy =np.uint8(np.absolute(sobely))
        sobel = cv2.bitwise_or(absx,absy)
        print("sobel")
        plt.imshow(sobel,'gray')
        plt.title('sobel gradient image')
        plt.show()





        kernel = np.ones((5,5),np.float32)/25
        smoothed = cv2.filter2D(sobel,-1,kernel)
        print("smoothed sobel")
        plt.imshow(smoothed,'gray')
        plt.title('smoothed sobel image')
        plt.show()




        sharp = np.uint8(np.absolute(sharpened))
        smooth = np.uint8(np.absolute(smoothed))
        mask=cv2.bitwise_and(sharp,smooth)
        print("mask")
        plt.imshow(mask,'gray')
        plt.title('mask filter image')
        plt.show()



        filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        s1=cv2.filter2D(img,-1,filter)
        s2=cv2.filter2D(mask,-1,filter)
        sharpened_image2=cv2.add(img,mask)

        plt.subplot(1,2,1)
        plt.imshow(img,'gray')
        plt.title('Previous image')
        plt.subplot(1,2,2)
        plt.imshow(sharpened_image2,'gray')
        plt.title('Sharpened image+mask filter')
        plt.show()




        gamma=0.5
        final_image = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
        plt.subplot(1,2,1)
        plt.imshow(img,'gray')
        plt.title('Previous image')
        plt.subplot(1,2,2)
        plt.imshow(final_image,'gray')
        plt.title('Final image power transform')
        plt.show() 
    else:
    

        image = mpimg.imread('https://drive.google.com/uc?export=view&id=1gGEYCY1mWDbkdmPW6-K-GjCzRiZmSaA6',cv2.IMREAD_COLOR)
        print("Original Photo")

        plt.figure(figsize=(7,7))

        plt.imshow(image,cmap='gray')
        plt.show()

        image = GaussianBlur(image)
    #     print("Blurred image")
    #     plt.imshow(image,cmap='gray')
    #     plt.show()
    #     print("Gradient intensity using sobel filters")
        image,angles = SobelFilter(image)
    #     plt.imshow(image,cmap='gray')
    #     plt.show()
    #     print("result of non -maximum supression")
        image = non_maximum_suppression(image, angles)
    #     plt.imshow(image,cmap='gray')
    #     plt.show()
    #     print("Threshold result")
        image = double_threshold_hysteresis(image,0.09,0.17)
    #     plt.imshow(image,cmap='gray')
    #     plt.show()
        print("Final result")
        image=hysteresis(image,75)
        plt.figure(figsize=(7,7))

        plt.imshow(image, 'gray')
        plt.show()
    #     visualize(image,'gray')
    

    
