#ILPF-Ideal high pass filter
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('highpass1.jpg',0)
row,col=img.shape
print('the row and cols values',row,col)
f = np.fft.fft2(img)
M1,N1=row/2,col/2
M1=int(M1)
N1=int(N1)       
print('the row and cols values',M1,N1)
#print('the values of are',f)
fshift = np.fft.fftshift(f)# to center zero freq coefficient
#print('the values of fshift are',fshift)
magnitude_spectrum = 20*np.log(np.abs(fshift))# to brighten display
D0=input('D0 value')
D0=int(D0)
dist_matrix=np.zeros(img.shape,np.uint8)
H_filter=dist_matrix
for i in range(row):
	for j in range(col):
                   dist_matrix[i][j]= math.sqrt((i-M1)**2+(j-N1)**2)

print(dist_matrix.shape)
for u in range(M1):
	for v in range(N1):
                   if(dist_matrix[u][v]>D0):
                                H_filter[u][v]=0
                                
                   else                     :
                                H_filter[u][v]=1
                   
print(H_filter.shape)
g=np.multiply(H_filter,f)
#print(g)
f_ishift = np.fft.ifftshift(g)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum1'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back, cmap = 'gray')
plt.title('idealsharpenedimage'), plt.xticks([]), plt.yticks([])
plt.show()




