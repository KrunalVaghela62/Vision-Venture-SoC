import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import requests

vid = requests.get("https://www.dropbox.com/s/f194zeyqbr00cjm/sample_video.mp4?dl=1").content
with open('sample_video.mp4', 'wb') as handler:
    handler.write(vid)
phoenix_image=cv2.imread("phoenix.jpg")

if phoenix_image is None:
    raise Exception("The image was not found! Check that you can see it on colab's file explorer by clicking the files icon.")
phoenix_image_rgb = cv2.cvtColor(phoenix_image, cv2.COLOR_BGR2RGB)

def imshow(image):
    if len(image.shape)==3:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    else:
        image= cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
phoenix_gray = cv2.cvtColor(phoenix_image, cv2.COLOR_BGR2GRAY)
arr=np.zeros(phoenix_gray.shape,dtype=np.uint8)
magenta_phoenix=np.stack([phoenix_gray,arr,phoenix_gray],axis=2)
print("created image of the shape ",magenta_phoenix.shape)
height,width,num_channel=magenta_phoenix.shape
bigger_phoenix=cv2.resize(magenta_phoenix,(width*3,height*2))




def crop_frame(frame,crop_size):
    crop_h,crop_w=crop_size
    cropped=frame[:crop_h,:crop_w]
    return cropped
capture=cv2.VideoCapture('sample_video.mp4')
output_path='output_cropped.mp4'
output_format=cv2.VideoWriter_fourcc('m','p','4','v')
output_fps=30
crop_size=(600,400)
cropped_output=cv2.VideoWriter(output_path,output_format,output_fps,crop_size)
n=0
while(True):
    boolean,next_frame=capture.read()
    if boolean==False:
        print("a total of" ,str(n), " frames have been captured ")
        break
    
    output_frame=crop_frame(next_frame,crop_size)
    cropped_output.write(output_frame)
    n=n+1
capture.release()
cropped_output.release()    


