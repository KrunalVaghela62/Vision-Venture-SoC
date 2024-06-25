#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


# < START >
arr = np.array([[1,2,4],[7,13,21]])
# < END >

print(arr)
print("Shape:", arr.shape)


# In[6]:


n_rows = 2
n_columns = 3

# 
x = np.random.normal(size=(n_rows,n_columns))
# 

print(x)


# In[9]:


# < START >
i=0
mylist=[]
for i in range(40):
    mylist.append(0)
    i=i+1
ARR=np.asarray(mylist)
ZERO_ARR=ARR.reshape(4,5,2)

# < END >

print(ZERO_ARR)

# < START >
# Initialize an array ONE_ARR of dimensions (4, 5, 2) whose every element is 1
i=0
mylist=[]
for i in range(40):
    mylist.append(1)
    i=i+1
ARR=np.asarray(mylist)
ONE_ARR=ARR.reshape(4,5,2)


# < END >

print(ONE_ARR)


# In[23]:


y = np.array([[1, 2, 3],
              [4, 5, 6]])

# < START >
# Create a new array y_transpose that is the transpose of matrix y
y1=y[0]
y2=y[1]
y_transpose=np.stack((y1,y2),axis=1)
y_r=y.copy()
print(y_transpose)
# < END >



# < START >
# Create a new array y_flat that contains the same elements as y but has been flattened to a column array
y_flat=y.reshape(-1)

# < END >

print(y_flat)


# In[24]:


# 
# Initialize the column matrix here
y=np.array([[4],[7],[11]])

# 

assert y.shape == (3, 1)
# The above line is an assert statement, which halts the program if the given condition evaluates to False.
# Assert statements are frequently used in neural network programs to ensure our matrices are of the right dimensions.

print(y)

# 
# Multiply both the arrays here
z=np.dot(y_r,y)
# 

assert z.shape == (2, 1)

print(z)


# In[17]:


x = np.array([4, 1, 5, 6, 11])

# 
# Create a new array y with the middle 3 elements of x
y=x[1:4]
# 

print(y)

z = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 
# Create a new array w with alternating elements of z
w=z[0::2]
# 

print(w)


# In[19]:


arr_2d = np.array([[4, 5, 2],
          [3, 7, 9],
          [1, 4, 5],
          [6, 6, 1]])

# 
# Create a 2D array sliced_arr_2d that is of the form [[5, 2], [7, 9], [4, 5]]
sliced_arr_2d=arr_2d[0:3,1:]

# 

print(sliced_arr_2d)
     


# In[26]:


arr1 = np.array([1, 2, 3, 4])
b = 1
arr1=arr1+b

# 
# Implement broadcasting to add b to each element of arr1

# 

print(arr1)

arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])
arr3 = np.array([[4],
                 [5]])

# 
# Multiply each element of the first row of arr2 by 4 and each element of the second row by 5, using only arr2 and arr3
arr2=arr2*arr3

# 

print(arr2)


# In[33]:


import time

arr_nonvectorized = np.random.rand(1000, 1000)
arr_vectorized = np.array(arr_nonvectorized) # making a deep copy of the array

start_nv = time.time()

# Non-vectorized approach
i=0
j=0
for i in range(1000):
    for j in range(1000):
        arr_nonvectorized[i,j]=arr_nonvectorized[i,j]*3
        j=j+1
    i=i+1





# 

end_nv = time.time()
print("Time taken in non-vectorized approach:", 1000*(end_nv-start_nv), "ms")

# uncomment and execute the below line to convince yourself that both approaches are doing the same thing
# print(arr_nonvectorized)

start_v = time.time()

# Vectorized approach
arr_vectorized =arr_vectorized *3


# 

end_v = time.time()
print("Time taken in vectorized approach:", 1000*(end_v-start_v), "ms")

# uncomment and execute the below line to convince yourself that both approaches are doing the same thing
# print(arr_vectorized)


# In[ ]:




