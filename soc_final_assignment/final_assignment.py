
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy import angle, real
from numpy import exp, abs, pi, sqrt
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
import requests

def imshow(im, cmap='gray'):
    # clip image from 0-1
    im = np.clip(im, 0, 1)
    plt.imshow(im, cmap=cmap)
imSizwe=9
magnification_factor=4
im1=np.zeros([imSizwe,imSizwe])
im2=np.zeros([imSizwe,imSizwe])
im1[0,0]=1
im2[0,1]=1
ff1 = fftshift(fft2(im1))
ff2 = fftshift(fft2(im2))

plt.figure()
plt.subplot(121)
imshow(im1)
plt.subplot(122)
imshow(im2)

plt.figure()
plt.subplot(121)
imshow(angle(ff1))
plt.subplot(122)
imshow(angle(ff2))
plt.figure()
plt.subplot(121)
imshow(abs(ff1))
plt.subplot(122)
imshow(abs(ff2))
def magnifyChange(im1, im2, magnificationFactor):

    # find phase shift in frequency domain
    im1Dft = fft2(im1)
    im2Dft = fft2(im2)
    phaseShift=np.angle(im1)-np.angle(im2Dft)
    New_Phase = phaseShift * magnificationFactor

    # Reconstruct the magnified image
    magnifiedDft = np.abs(im2Dft) * np.exp(1j * New_Phase)
    # what does the magnified phase change cause in image space?

    magnified = ifft2(magnifiedDft).real

    return magnified
# magnify position change
magnified = magnifyChange(im1, im2, magnification_factor)

plt.figure(figsize=(12,36))
plt.subplot(131)
imshow(im1)
plt.title('im1')

plt.subplot(132)
imshow(im2)
plt.title('im2')

plt.subplot(133)
imshow(magnified) 
plt.title('magnified')
plt.savefig("problem_3a.png")
# Define the image size and magnification factor.
imSize = 9
magnificationFactor = 4

# Create the two images.
im1 = np.zeros([imSize, imSize])
im2 = np.zeros([imSize, imSize])
im1[0,0] = 1
im2[0,1] = 1
im1[8,8] = 1
im2[7,8] = 1

# Manually edit the expected matrix (currently set as zeros) by creating 1s to show the expected output.
expected = np.zeros([imSize, imSize])
expected[0, magnificationFactor] = 1
expected[8 - magnificationFactor, 8] = 1

# Magnify the position change.
magnified = magnifyChange(im1, im2, magnificationFactor)

# Plot the images.
plt.figure(figsize=(12,36))
plt.subplot(141)
plt.imshow(im1, cmap='gray', interpolation='nearest')
plt.title('im1')
plt.subplot(142)
plt.imshow(im2, cmap='gray', interpolation='nearest')
plt.title('im2')
plt.subplot(143)
plt.imshow(expected, cmap='gray', interpolation='nearest')
plt.title('expected')
plt.subplot(144)
plt.imshow(magnified, cmap='gray', interpolation='nearest')
plt.title('magnified')
plt.tight_layout()
plt.savefig("problem_3b.png")
import scipy.signal as signal
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt


vid = requests.get("http://people.csail.mit.edu/mrub/evm/video/baby.mp4").content
with open('sbaby.mp4', 'wb') as handler:
    handler.write(vid)

