#!/usr/bin/env python
# coding: utf-8

# In[7]:


class Myclass:
    x=5
Ob=Myclass()

print(Ob.x)


# In[4]:


class Myclass:


# In[8]:


class person:


# In[72]:


class person:
    def __init__(s,name,age):
        s.name=name
        s.name=age
    def myfunc(p):
        print( p.name," ",p.age)
p1=person("john",str(36))
p1.myfunc()


# In[32]:


p1.myfunc()


# In[64]:


class student(person):
     def __init__(s,name,age,year):
            super().__init__(name,age)
            s.year=year     
     def newfunc(s):
        print("my name is "+s.name+",my age is "+s.age+",my year of graduation is "+s.year)


# In[65]:


x=student("mark",str(45),str(2019))


# In[66]:


x.newfunc()


# In[69]:


mytuple="nigga"
myit=iter(mytuple)
print(next(myit))
print(next(myit))
print(next(myit))


# In[70]:


class mynumbers:
    def __iter__(s):
        s.a=1
        print(a)
no=mynumbers()


# In[74]:


class mynumber:
    x="apple"
    y="banana"
    z="mango"
clas=mynumber()
myit=iter(clas)


# In[77]:


class mynumber:
    def __iter__(self):
        return self
    def __next__(self):
        self.a=1
        x=self.a
        self.a+=1
        return x
num=mynumber()
myit=iter(num)
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))


# In[ ]:




