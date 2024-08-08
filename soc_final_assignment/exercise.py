#exercsie 1
num_students = 4
num_assignments = 5

# Write your solution below.

import numpy as np
import matplotlib.pyplot as plt
grade=np.arange(0,num_students*num_assignments).reshape(num_students,num_assignments)

julie_grades=grade[2,0:]

PSset4=grade[0:,4]

#exercise 2
num_keypoints = 7
num_joints = 5


# All Z's in one plane, but makes it easier to see XYZ vs Start/end
keypoint_positions = np.array(
    [
        [0, 1,0], #Head
        [0, 0,0], #Torso
        [1, 0,0], #Right Arm
        [-1, 0,0], #Left Arm
        [0, -1,0], #Lower ,Torso
        [1, -2,0], #Right Leg
        [-1, -2,0] #Left Leg
    ]
)

#   O
#  _|_
#   |
#  /\
joints = np.array([
    # Head to torso
    [0, 1],
    # Torso to Right arm
    [1, 2],
    # Torso to Left Arm
    [1, 3],
    # Torso to Lower Torso
    [3, 4],
    # Lower Torso to Right Leg
    [4, 5],
    # Lower Torso to Left Leg
    [4, 6]
])
#write your solution below
joint_starts=np.array([keypoint_positions[0,0:],keypoint_positions[1,0:],keypoint_positions[1,0:],keypoint_positions[3,0:],keypoint_positions[4,0:],keypoint_positions[4,0:]])
joint_end=np.array([keypoint_positions[1,0:],keypoint_positions[2,0:],keypoint_positions[3,0:],keypoint_positions[4,0:],keypoint_positions[5,0:],keypoint_positions[6,0:]])
joint_displacement=np.array([keypoint_positions[1,0:]-keypoint_positions[0,0:],keypoint_positions[2,0:]-keypoint_positions[1,0:],keypoint_positions[3,0:]-keypoint_positions[1,0:],keypoint_positions[4,0:]-keypoint_positions[3,0:],keypoint_positions[5,0:]-keypoint_positions[4,0:],keypoint_positions[6,0:]-keypoint_positions[4,0:]])
arr_z=np.array([0]*(num_joints))
for i in range(num_joints):
    arr_z[i]=((joint_displacement[i,0])**2+(joint_displacement[i,1])**2+(joint_displacement[i,2])**2)**0.5
print(arr_z)
arr=np.array([0]*(num_joints*2))
mat_arr=arr.reshape(num_joints,2)
arr1=np.array([0]*(num_joints*2))
mat_arr1=arr.reshape(num_joints,2)
for j in range(num_joints):
    for k in range(2):
        mat_arr[j,0]=joint_starts[j,0]
        mat_arr[j,1]=joint_end[j,0]
for i in range(num_joints):
    for p in range(2):
        mat_arr1[i,0]=joint_starts[i,1]
        mat_arr1[i,1]=joint_end[i,1]



for l in range(num_joints):
    plt.plot(mat_arr[l,0:],mat_arr1[l,0:])

#exercise 3
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.signal
conv2d = scipy.signal.convolve2d # assigning a shorter name for this function.

# looks for horizontal edges
horizontal_edge_detector = np.array(
  [
      [-1, 0, 1]
  ]
)

box_blur_size = 15
box_blur = np.ones((box_blur_size, box_blur_size)) / (box_blur_size ** 2)
box_blur1=box_blur.reshape(box_blur_size,box_blur_size)
sharpen_kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0,  -1, 0]
    ]
)

all_edge_detector = np.array(
    [
        [0, -1, 0],
        [-1, 4, -1],
        [0,  -1, 0]
    ]
)

def prep_to_draw(img):
  """ Function which takes in an image and processes it to display it.
  """
  # Scale to 0,255
  prepped = img * 255
  # Clamp to [0, 255]
  prepped = np.clip(prepped, 0, 255) # clips values < 0 to 0 and > 255 to 255.
  prepped = prepped.astype(np.uint8)
  return prepped
phoenix_image=cv2.imread("phoenix.jpg")
phoenix_grayscale=cv2.cvtColor(phoenix_image,cv2.COLOR_BGR2GRAY)
def imshow(image):
    if len(image.shape)==3:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    else:
        image= cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

imshow(prep_to_draw(conv2d(phoenix_grayscale,box_blur1)))










