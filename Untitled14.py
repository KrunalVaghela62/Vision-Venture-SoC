#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy


# In[2]:


print(numpy.__version__)


# In[27]:


import numpy as np
arr=np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr[0:2,2])


# In[24]:


arr.ndmin


# In[46]:


import numpy as np

arr = np.array([1, 2, 3, 4,5,6,7,8,9,10,11,12])
newarr=arr.reshape(2,3,2)
x=newarr.view()
x[0,2,0]=123
print(newarr)


# In[61]:


import numpy as np
arr=np.array([[1,2,3,4],[5,6,7,8]])
for index,x in np.ndenumerate(arr):
    print(index,x)
    


# In[62]:


import numpy as np
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
arr=np.concatenate((arr1,arr2))
print(arr)


# In[ ]:




