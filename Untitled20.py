#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
def myadd(x,y):
    return x+y
myadd=np.frompyfunc(myadd,2,1)
print(myadd([1,2,3,4,5],[6,7,8,9,10]))
print(type(np.add))


# In[4]:


print(type(np.concatenate))


# In[12]:


import numpy as np
arr = np.around(3.8777)
print(arr)


# In[16]:


import numpy as np
arr=np.arange(1,10)
print(np.log(arr))


# In[ ]:


import numpy as np
arr=np.array([1,2,3,4,5])
x=np.cumsum(arr)
print(x)

