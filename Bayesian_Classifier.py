# -*- coding: utf-8 -*-
"""

@author: Ayush Dwivedi
"""



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

cols=['x1','x2']
df1=pd.read_csv("l_class1.txt",header=None,sep=" ",names=cols)
df2=pd.read_csv("l_class2.txt", header=None,names=cols,sep=" ")
df3=pd.read_csv("l_class3.txt",sep=" ",names=cols,header=None)


def bivariate(X,mean,co_var):
    z= []
    d = abs(np.linalg.det(co_var))
    e = math.sqrt(d)
    for x in X:
       
        prd1 = np.dot((x-mean).transpose(),np.linalg.inv(co_var))
        y = (1* np.exp(np.dot(prd1,(x-mean))))/(2* math.pi * e)  
        z.append(y)
    return z

#Finding the training and testing data.
df1=df1.sample(frac=1)
df2=df2.sample(frac=1)
df3=df3.sample(frac=1)
X1=np.array(df1)
X2=np.array(df2)
X3=np.array(df3)
n_train = math.floor(0.5 * X1.shape[0])
n_test = math.ceil(0.5 * X1.shape[0])
X1_train = X1[:n_train]
X1_test = X1[n_train:]

X2_train = X2[:n_train]
X2_test = X2[n_train:]

X3_train = X3[:n_train]
X3_test = X3[n_train:]



#Calculating Mean and Variance for Class 1.
a=0
mu1=[]
s1=[]
for i in X1_train[:,0]:
    a=a+i
mean1=a/250
mu1.append(mean1)
a=0
for i in X1_train[:,0]:
    a=a+(i-mean1)**2
variance1=a/249
s1.append(variance1)
a=0
for i in X1_train[:,1]:
    a=a+i
mean2=a/250
mu1.append(mean2)
a=0
for i in X1_train[:,1]:
    a=a+(i-mean1)**2
variance2=a/249
s1.append(variance2)
a=0
for i in range(250):
    a=a+((X1_train[i][0]-mean1)*(X1_train[i][1]-mean2))
Cov11=a/249

#Calculating Mean and Variance for Class 2.
a=0
mu2=[]
s2=[]
mean1=0
mean2=0
variance1=0
variance2=0
for i in X2_train[:,0]:
    a=a+i
mean1=a/250
mu2.append(mean1)
a=0
for i in X2_train[:,0]:
    a=a+(i-mean1)**2
variance1=a/249
s2.append(variance1)
a=0
for i in X2_train[:,1]:
    a=a+i
mean2=a/250
mu2.append(mean2)
a=0
for i in X2_train[:,1]:
    a=a+(i-mean2)**2
variance2=a/249 
s2.append(variance2)
a=0
for i in range(250):
    a=a+((X2_train[i][0]-mean1)*(X2_train[i][1]-mean2))
Cov12=a/249

#Calculating Mean and Variance for Class 3.
a=0
mu3=[]
s3=[]
mean1=0
mean2=0
variance1=0
variance2=0
for i in X3_train[:,0]:
    a=a+i
mean1=a/250
mu3.append(mean1)
a=0
for i in X3_train[:,0]:
    a=a+(i-mean1)**2
    
variance1=a/249
s3.append(variance1)
a=0
for i in X3_train[:,1]:
    a=a+i
mean2=a/250

mu3.append(mean2)
a=0
for i in X3_train[:,1]:
    a=a+(i-mean2)**2
variance2=a/249
s3.append(variance2)
a=0
for i in range(250):
    a=a+((X3_train[i][0]-mean1)*(X3_train[i][1]-mean2))
Cov13=a/249 

#plt.scatter(, y, kwargs)

#For forming C1 classifier first finding covariance matrix.
from scipy.stats import multivariate_normal as mn
GT=[]
for i in range(750):
    if 0<=i<250:
        GT.append(1)
    elif 249<i<500:
        GT.append(2)
    else:
        GT.append(3)     
X_test=np.concatenate((X1_test,X2_test,X3_test),axis=0)
M1=[]
T1=[]
a=0
for i in range(2):
    a=a+s1[i]+s2[i]+s3[i]
sig1=a/6    
Cov1=np.array([[sig1,0],[0,sig1]])
p1=list(mn.pdf(X_test,mean=mu1,cov=Cov1))
p2=list(mn.pdf(X_test,mean=mu2,cov=Cov1))
p3=list(mn.pdf(X_test,mean=mu3,cov=Cov1))
for i in range(750):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(750):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
        
plt.scatter(X_test[:,0], X_test[:,1],c=T1)       
def Tab(T,M):
    a=0        
    for i in range(750):
        if T[i]==M[i]:
            a=a+1
    A1=(a/750)*100


     
    #All values for the table.
    P1=[]
    R1=[]
    F1=[]
    c=0
    tn=0
    tp=0
    fn=0
    fp=0
    for k in T:
        
        if (k==1) and (M[c]==1):
            tp=tp+1
        elif (k==1) and ((M[c]==2) or (M[c]==3)):
            fn=fn+1
        elif ((k==2) or (k==3)) and (M[c]==1):
            fp=fp+1
        elif (k!=1) and (M[c]!=1):
            tn=tn+1
        c=c+1
    p11=tp/(tp+fp)
    r11=tp/(tp+fn)

    P1.append(p11)
    R1.append(r11)
    F1.append(2*(p11*r11)/(p11+r11))





    c=0
    tn=0
    tp=0
    fn=0
    fp=0          
    for k in T:
        
        if (k==2) and (M[c]==2):
            tp=tp+1
        elif (k==2) and ((M[c]==1) or (M[c]==3)):
            fn=fn+1
        elif ((k==1) or (k==3)) and (M[c]==2):
            fp=fp+1
        elif (k!=2) and (M[c]!=2):
            tn=tn+1
        c=c+1
    p11=tp/(tp+fp)
    r11=tp/(tp+fn)

    P1.append(p11)
    R1.append(r11)
    F1.append(2*(p11*r11)/(p11+r11))          
              
    c=0
    tn=0
    tp=0
    fn=0
    fp=0          
    for k in T:
        
        if (k==3) and (M[c]==3):
            tp=tp+1
        elif (k==3) and ((T1[c]==1) or (M[c]==2)):
            fn=fn+1
        elif ((k==1) or (k==2)) and (M[c]==3):
            fp=fp+1
        elif (k!=3) and (M[c]!=3):
            tn=tn+1
        c=c+1
    p11=tp/(tp+fp)
    r11=tp/(tp+fn)

    P1.append(p11)
    R1.append(r11)
    F1.append(2*(p11*r11)/(p11+r11)) 
    p11=(P1[0]+P1[1]+P1[2])/3
    r11=(R1[0]+R1[1]+R1[2])/3
    f11=(F1[0]+F1[1]+F1[2])/3        
    C1=[A1,p11,r11,f11]
    return C1 
C11=Tab(GT, T1) 
print("TABLE FOR LINEAR SEPARABLE DATA\n")        
print("Classifier","Accuracy","Precision","Recall","F-score",sep="        ") 
print("C1",round(C11[0],2),round(C11[1],2),round(C11[2],2),round(C11[3],2),sep="               ")  



D1=[]
for i in range(-30,30):
    for j in range(-30,30):
        D1.append([i,j])
M1=[]
T1=[]
p1=list(mn.pdf(D1,mean=mu1,cov=Cov1))
p2=list(mn.pdf(D1,mean=mu2,cov=Cov1))
p3=list(mn.pdf(D1,mean=mu3,cov=Cov1))

for i in range(3600):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(3600):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
D11=np.array(D1)
xx,yy=np.meshgrid(D11[:,0],D11[:,1],sparse=True)
       
plt.scatter(xx, yy,c=T1,cmap="Set3") 

plt.show()

#Finding C2 classifier.
M1=[]
T1=[]  
Cov1=np.array([[(s1[0]+s2[0]+s3[0])/3,(Cov11+Cov12+Cov13)/3],[(Cov11+Cov12+Cov13)/3,(s1[1]+s2[1]+s3[1])/3]])
p1=list(mn.pdf(X_test,mean=mu1,cov=Cov1))
p2=list(mn.pdf(X_test,mean=mu2,cov=Cov1))
p3=list(mn.pdf(X_test,mean=mu3,cov=Cov1))
for i in range(750):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(750):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)

#All values for the table C2.
C11=Tab(GT, T1)
print("C2",round(C11[0],2),round(C11[1],2),round(C11[2],2),round(C11[3],2),sep="               ")
         
M1=[]
T1=[]
p1=list(mn.pdf(D1,mean=mu1,cov=Cov1))
p2=list(mn.pdf(D1,mean=mu2,cov=Cov1))
p3=list(mn.pdf(D1,mean=mu3,cov=Cov1))

for i in range(3600):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(3600):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
xx,yy=np.meshgrid(D11[:,0],D11[:,1],sparse=True)
       
plt.scatter(xx, yy,c=T1,cmap="Set3") 

plt.show()

#Finding C3 classifier.
M1=[]
T1=[]
a=0
for i in range(2):
    a=a+s1[i]+s2[i]+s3[i]
sig1=a/6    
Cov1=np.array([[s1[0],0],[0,s1[1]]])
Cov2=np.array([[s2[0],0],[0,s2[1]]])
Cov3=np.array([[s3[0],0],[0,s3[1]]])               
p1=list(mn.pdf(X_test,mean=mu1,cov=Cov1))
p2=list(mn.pdf(X_test,mean=mu2,cov=Cov2))
p3=list(mn.pdf(X_test,mean=mu3,cov=Cov3))
for i in range(750):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(750):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
C11=Tab(GT, T1)
print("C3",round(C11[0],2),round(C11[1],2),round(C11[2],2),round(C11[3],2),sep="               ")

M1=[]
T1=[]
p1=list(mn.pdf(D1,mean=mu1,cov=Cov1))
p2=list(mn.pdf(D1,mean=mu2,cov=Cov2))
p3=list(mn.pdf(D1,mean=mu3,cov=Cov3))

for i in range(3600):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(3600):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
xx,yy=np.meshgrid(D11[:,0],D11[:,1],sparse=True)
       
plt.scatter(xx, yy,c=T1,cmap="Set3") 

plt.show()

#Finding C4 classifier.
M1=[]
T1=[]   
Cov1=np.array([[s1[0],Cov11],[Cov11,s1[1]]])
Cov2=np.array([[s2[0],Cov12],[Cov12,s2[1]]])
Cov3=np.array([[s3[0],Cov13],[Cov13,s3[1]]])               
p1=list(mn.pdf(X_test,mean=mu1,cov=Cov1))
p2=list(mn.pdf(X_test,mean=mu2,cov=Cov2))
p3=list(mn.pdf(X_test,mean=mu3,cov=Cov3))
for i in range(750):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(750):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)

C11=Tab(GT, T1)
print("C4",round(C11[0],2),round(C11[1],2),round(C11[2],2),round(C11[3],2),sep="               ")

M1=[]
T1=[]
p1=list(mn.pdf(D1,mean=mu1,cov=Cov1))
p2=list(mn.pdf(D1,mean=mu2,cov=Cov2))
p3=list(mn.pdf(D1,mean=mu3,cov=Cov3))

for i in range(3600):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(3600):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
xx,yy=np.meshgrid(D11[:,0],D11[:,1],sparse=True)
       
plt.scatter(xx, yy,c=T1,cmap="Set3") 

plt.show()

#For Non Linear Classes.

df4=pd.read_csv("nl_class1.txt",header=None,sep=" ",names =['0','1','2'])
df5=pd.read_csv("nl_class2.txt", header=None,sep=" ",names =['0','1','2'])
df6=pd.read_csv("nl_class3.txt",sep=" ",header=None,names =['0','1','2'])
#df41=df4.dropna()
del df4['2'] 
del df5['2']
del df6['2'] 
df4=df4.sample(frac=1)
df5=df5.sample(frac=1)
df6=df6.sample(frac=1)
X1=np.array(df4)
X2=np.array(df5)
X3=np.array(df6)
n_train = math.floor(0.5 * X1.shape[0])
n_test = math.ceil(0.5 * X1.shape[0])
X1_train = X1[:n_train]
X1_test = X1[n_train:]

X2_train = X2[:n_train]
X2_test = X2[n_train:]

X3_train = X3[:n_train]
X3_test = X3[n_train:]


#Calculating Mean and Variance for Class 1.
a=0
mu1=[]
s1=[]
for i in X1_train[:,0]:
    a=a+i
mean1=a/250
mu1.append(mean1)
a=0
for i in X1_train[:,0]:
    a=a+(i-mean1)**2
variance1=a/249
s1.append(variance1)
a=0
for i in X1_train[:,1]:
    a=a+i
mean2=a/250
mu1.append(mean2)
a=0
for i in X1_train[:,1]:
    a=a+(i-mean1)**2
variance2=a/249
s1.append(variance2)
a=0
for i in range(250):
    a=a+((X1_train[i][0]-mean1)*(X1_train[i][1]-mean2))
Cov11=a/249    
    
    
     
    
#Calculating Mean and Variance for Class 2.
a=0
mu2=[]
s2=[]
mean1=0
mean2=0
variance1=0
variance2=0
for i in X2_train[:,0]:
    a=a+i
mean1=a/250
mu2.append(mean1)
a=0
for i in X2_train[:,0]:
    a=a+(i-mean1)**2
variance1=a/249
s2.append(variance1)
a=0
for i in X2_train[:,1]:
    a=a+i
mean2=a/250
mu2.append(mean2)
a=0
for i in X2_train[:,1]:
    a=a+(i-mean2)**2
variance2=a/249 
s2.append(variance2)
a=0
for i in range(250):
    a=a+((X2_train[i][0]-mean1)*(X2_train[i][1]-mean2))
Cov12=a/249 
       

#Calculating Mean and Variance for Class 3.
a=0
mu3=[]
s3=[]
mean1=0
mean2=0
variance1=0
variance2=0
for i in X3_train[:,0]:
    a=a+i
mean1=a/250
mu3.append(mean1)
a=0
for i in X3_train[:,0]:
    a=a+(i-mean1)**2
    
variance1=a/249
s3.append(variance1)
a=0
for i in X3_train[:,1]:
    a=a+i
mean2=a/250

mu3.append(mean2)
a=0
for i in X3_train[:,1]:
    a=a+(i-mean2)**2
variance2=a/249
s3.append(variance2)
a=0
for i in range(250):
    a=a+((X3_train[i][0]-mean1)*(X3_train[i][1]-mean2))
Cov13=a/249

#For forming C1 classifier first finding covariance matrix.
from scipy.stats import multivariate_normal as mn
GT=[]
for i in range(750):
    if 0<=i<250:
        GT.append(1)
    elif 249<i<500:
        GT.append(2)
    else:
        GT.append(3)     
X_test=np.concatenate((X1_test,X2_test,X3_test),axis=0)
M1=[]
T1=[]
a=0
for i in range(2):
    a=a+s1[i]+s2[i]+s3[i]
sig1=a/6    
Cov1=np.array([[sig1,0],[0,sig1]])
p1=list(mn.pdf(X_test,mean=mu1,cov=Cov1))
p2=list(mn.pdf(X_test,mean=mu2,cov=Cov1))
p3=list(mn.pdf(X_test,mean=mu3,cov=Cov1))
for i in range(750):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(750):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
        
C11=Tab(GT, T1) 
print("TABLE FOR NON-LINEAR SEPARABLE DATA\n")        
print("Classifier","Accuracy","Precision","Recall","F-score",sep="        ") 
print("C1",round(C11[0],2),round(C11[1],2),round(C11[2],2),round(C11[3],2),sep="               ")  

M1=[]
T1=[]
p1=list(mn.pdf(D1,mean=mu1,cov=Cov1))
p2=list(mn.pdf(D1,mean=mu2,cov=Cov2))
p3=list(mn.pdf(D1,mean=mu3,cov=Cov3))

for i in range(3600):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(3600):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
xx,yy=np.meshgrid(D11[:,0],D11[:,1],sparse=True)
       
plt.scatter(xx, yy,c=T1,cmap="Set3") 

plt.show()

#Finding C2 classifier.
M1=[]
T1=[]  
Cov1=np.array([[(s1[0]+s2[0]+s3[0])/3,(Cov11+Cov12+Cov13)/3],[(Cov11+Cov12+Cov13)/3,(s1[1]+s2[1]+s3[1])/3]])
p1=list(mn.pdf(X_test,mean=mu1,cov=Cov1))
p2=list(mn.pdf(X_test,mean=mu2,cov=Cov1))
p3=list(mn.pdf(X_test,mean=mu3,cov=Cov1))
for i in range(750):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(750):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)

#All values for the table C2.
C11=Tab(GT, T1)
print("C2",round(C11[0],2),round(C11[1],2),round(C11[2],2),round(C11[3],2),sep="               ")
         
M1=[]
T1=[]
p1=list(mn.pdf(D1,mean=mu1,cov=Cov1))
p2=list(mn.pdf(D1,mean=mu2,cov=Cov2))
p3=list(mn.pdf(D1,mean=mu3,cov=Cov3))

for i in range(3600):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(3600):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
xx,yy=np.meshgrid(D11[:,0],D11[:,1],sparse=True)
       
plt.scatter(xx, yy,c=T1,cmap="Set3") 

plt.show()

#Finding C3 classifier.
M1=[]
T1=[]
a=0
for i in range(2):
    a=a+s1[i]+s2[i]+s3[i]
sig1=a/6    
Cov1=np.array([[s1[0],0],[0,s1[1]]])
Cov2=np.array([[s2[0],0],[0,s2[1]]])
Cov3=np.array([[s3[0],0],[0,s3[1]]])               
p1=list(mn.pdf(X_test,mean=mu1,cov=Cov1))
p2=list(mn.pdf(X_test,mean=mu2,cov=Cov2))
p3=list(mn.pdf(X_test,mean=mu3,cov=Cov3))
for i in range(750):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(750):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
C11=Tab(GT, T1)
print("C3",round(C11[0],2),round(C11[1],2),round(C11[2],2),round(C11[3],2),sep="               ")

M1=[]
T1=[]
p1=list(mn.pdf(D1,mean=mu1,cov=Cov1))
p2=list(mn.pdf(D1,mean=mu2,cov=Cov2))
p3=list(mn.pdf(D1,mean=mu3,cov=Cov3))

for i in range(3600):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(3600):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
xx,yy=np.meshgrid(D11[:,0],D11[:,1],sparse=True)
       
plt.scatter(xx, yy,c=T1,cmap="Set3") 

plt.show()

#Finding C4 classifier.
M1=[]
T1=[]   
Cov1=np.array([[s1[0],Cov11],[Cov11,s1[1]]])
Cov2=np.array([[s2[0],Cov12],[Cov12,s2[1]]])
Cov3=np.array([[s3[0],Cov13],[Cov13,s3[1]]])               
p1=list(mn.pdf(X_test,mean=mu1,cov=Cov1))
p2=list(mn.pdf(X_test,mean=mu2,cov=Cov2))
p3=list(mn.pdf(X_test,mean=mu3,cov=Cov3))
for i in range(750):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(750):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)

C11=Tab(GT, T1)
print("C4",round(C11[0],2),round(C11[1],2),round(C11[2],2),round(C11[3],2),sep="               ")

M1=[]
T1=[]
p1=list(mn.pdf(D1,mean=mu1,cov=Cov1))
p2=list(mn.pdf(D1,mean=mu2,cov=Cov2))
p3=list(mn.pdf(D1,mean=mu3,cov=Cov3))

for i in range(3600):
    M1.append(max(p1[i],p2[i],p3[i]))
for i in range(3600):
    if M1[i]==p1[i]:
        T1.append(1)
    elif M1[i]==p2[i]:
        T1.append(2)
    else:
        T1.append(3)
xx,yy=np.meshgrid(D11[:,0],D11[:,1],sparse=True)
       
plt.scatter(xx, yy,c=T1,cmap="Set3") 

plt.show()