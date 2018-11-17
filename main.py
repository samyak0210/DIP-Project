import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import choice
from scipy import ndimage
import scipy
from PIL import Image
import random
from ransac import *
from seam import *
from tqdm import *

def main():
    im1 = cv2.imread('1.jpg', 0)
    im2 = cv2.imread('2.jpg', 0)
    im1_copy = im1.copy()
    im1 = cv2.copyMakeBorder(im1,200,200,500,500, cv2.BORDER_CONSTANT)
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(des1, des2)

    correspondenceList = []

    for m in matches:
        (x1, y1) = kp1[m.queryIdx].pt
        (x2, y2) = kp2[m.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    
    
    corrs = np.matrix(correspondenceList)
    out_arr, inliers = ransac(corrs, 5.0)
    print(len(out_arr))
    mi = np.inf
    final_image = None
    for i in trange(len(out_arr)):
        out_ransac = out_arr[i]
        out = cv2.warpPerspective(im2, scipy.linalg.inv(out_ransac), (im1.shape[1],  im1.shape[0]))
        output = np.zeros_like(im1)
        (x, y) = im1.shape
        seam = []
        for i in range(x):
            for j in range(y):
                if im1[i][j]==0 and out[i][j]==0:
                    output[i][j]=0
                elif im1[i][j]==0:
                    output[i][j] = out[i][j]
                elif out[i][j]==0:
                    output[i][j] = (im1[i][j])
                else:
                    output[i,j] = energy(im1, out, 128, 2, i, j)
                    seam.append([i,j])
                    # output[i][j] = (int(int(im1[i][j]) + int(out[i][j]))/2)
        error = 0
        for [x, y] in seam:
            error += np.sum((output[x-8:x+9, y-8:y+9] - im1[x-8:x+9, y-8:y+9])**2 + (output[x-8:x+9, y-8:y+9] - out[x-8:x+9, y-8:y+9])**2)
        if error/len(seam) < mi:
            print("hi")
            mi = error/len(seam)
            final_image = np.copy(output)


    plt.subplot(2,2,1)
    plt.axis('off')
    plt.imshow(im1_copy, cmap='gray')
    
    plt.subplot(2,2,2)
    plt.axis('off')
    plt.imshow(im2, cmap='gray')

    plt.subplot(2,2,3)
    plt.axis('off')
    plt.imshow(final_image, cmap='gray')

    plt.subplot(2,2,4)
    plt.axis('off')
    plt.imshow(out, cmap='gray')

    plt.show()

    cv2.imwrite('result1.jpg', final_image)

if __name__ == "__main__":
    main()
