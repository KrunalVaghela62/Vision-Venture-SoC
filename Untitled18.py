#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
arr1=np.array([[1,2,3],[4,5,6]])
arr2=np.array([[7,8,9],[10,11,12]])
arr=np.concatenate((arr1,arr2),axis=1)
print(arr)


# In[28]:


import numpy as np
arr1=np.array([[1,2,3],[4,5,6]])
arr2=np.array([[7,8,9],[10,11,12]])
arr=np.stack((arr1,arr2),axis=1)
print(arr)


# In[34]:


import numpy as np
arr1=np.array([1,2])
arr2=np.array([3,4])
arr=np.dstack((arr1,arr2))
print(arr)


# In[35]:


import numpy as np
arr=np.array([1,2,3,4,5,6,7])
newarr=np.array_split(arr,3)
print(newarr[0])
print(newarr[1])
print(newarr[2])


# In[50]:


import numpy as np
arr=np.array([1,3,5,7])
x=np.searchsorted(arr,[2,4,6])
print(x)


# In[45]:


import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3, axis=1)

print(newarr)


# In[51]:


import numpy as np
arr=np.array([1,2,3,4])
x=[True,False,True,False]
newarr=arr[x]
print(newarr)


# In[60]:


import numpy as np
arr=np.array([1,2,3,43,5,35,56,7,8,99])
filter_arr=arr%2==0
newarr=arr[filter_arr]
print(newarr)
print(filter_arr)


# In[89]:


from numpy import random
x=random.choice([1,2],p=[0.1,0.9],size=(3,5))
print(x)


# In[95]:


from numpy import random
import numpy as np
arr=np.array([1,2,3,4,5])
newarr=random.permutation(arr)
arr[0]=1234
print(newarr)
print(arr)


# In[96]:


get_ipython().system('pip install seaborn')


# In[97]:


import matplotlib.pyplot as plt



# In[98]:


get_ipython().system('pip install matplotlib')


# In[103]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot([0,1,2,3,4,5],hist=False)
plt.show()


# In[113]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.normal(size=1000),hist=False)
plt.show()


# In[120]:


from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.binomial(n=100,p=0.5,size=10000))
plt.show()


# In[ ]:




