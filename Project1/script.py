
# coding: utf-8

# In[3]:


# Team Member Name: Anupriya Goyal (anupriya) , Sagar Pokle (sagarpok), Sravanthi Adibhatla (sadibhat)

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
from scipy.stats import multivariate_normal 
#Claculating mean_vector 
def m_vector(X,y):
    mean_vect1 =[]
    mean_vect2 =[]
    mean_vect3 =[]
    mean_vect4 =[]
    mean_vect5 =[]
    for rw in range(1,len(X)):
            if(y[rw]==1.):
                list =[]
                for cl in range(len(X[0])):
                    list.append(X[rw][cl])
                mean_vect1.append(list)   
            
            if(y[rw]==2.):
                list =[]
                for cl in range(len(X[0])):
                    list.append(X[rw][cl])
                mean_vect2.append(list) 
                
            if(y[rw]==3.):
                list =[]
                for cl in range(len(X[0])):
                    list.append(X[rw][cl])
                mean_vect3.append(list) 
                
            if(y[rw]==4.):
                list =[]
                for cl in range(len(X[0])):
                    list.append(X[rw][cl])
                mean_vect4.append(list)   
                
            if(y[rw]==5.):
                list =[]
                for cl in range(len(X[0])):
                    list.append(X[rw][cl])
                mean_vect5.append(list) 
                
    a=np.array(mean_vect1)
    mean1=np.mean(a,axis=0)
    a=np.array(mean_vect2)
    mean2=np.mean(a,axis=0)
    a=np.array(mean_vect3)
    mean3=np.mean(a,axis=0)
    a=np.array(mean_vect4)
    mean4=np.mean(a,axis=0)
    a=np.array(mean_vect5)
    mean5=np.mean(a,axis=0)

    list1 =[]
    list1.append(mean1)
    list1.append(mean2)
    list1.append(mean3)
    list1.append(mean4)
    list1.append(mean5)
    
    means =np.array(list1)
    means=np.transpose(means)
    return means
def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 

    means = m_vector(X,y)
    S= X.shape
    c =np.array(X)
    a=np.std(c,axis=0)
    covmat=np.zeros([S[1],S[1]])
    for x in range(S[1]):
        covmat[x][x]=(a[x])**2
                
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    mean_vect1 =[]
    mean_vect2 =[]
    mean_vect3 =[]
    mean_vect4 =[]
    mean_vect5 =[]
    
    for rw in range(1,len(X)):
            if(y[rw]==1.):
                list =[]
                for cl in range(len(X[0])):
                    list.append(X[rw][cl])
                mean_vect1.append(list)   
            
            if(y[rw]==2.):
                list =[]
                for cl in range(len(X[0])):
                    list.append(X[rw][cl])
                mean_vect2.append(list) 
                
            if(y[rw]==3.):
                list =[]
                for cl in range(len(X[0])):
                    list.append(X[rw][cl])
                mean_vect3.append(list) 
                
            if(y[rw]==4.):
                list =[]
                for cl in range(len(X[0])):
                    list.append(X[rw][cl])
                mean_vect4.append(list)   
                
            if(y[rw]==5.):
                list =[]
                for cl in range(len(X[0])):
                    list.append(X[rw][cl])
                mean_vect5.append(list) 
    
    mean_vect1=np.array(mean_vect1)
    mean_vect2=np.array(mean_vect2)
    mean_vect3=np.array(mean_vect3)
    mean_vect4=np.array(mean_vect4)
    mean_vect5=np.array(mean_vect5) 
    
    #Transpose of mean_Vectors
    mean_vect1 = np.transpose(mean_vect1)
    mean_vect2 = np.transpose(mean_vect2)
    mean_vect3 = np.transpose(mean_vect3)
    mean_vect4 = np.transpose(mean_vect4)
    mean_vect5 = np.transpose(mean_vect5)
    
    covmats =[]
    cov1=np.cov(mean_vect1)
    covmats.append(cov1)
    cov2=np.cov(mean_vect2)
    covmats.append(cov2)
    cov3=np.cov(mean_vect3)
    covmats.append(cov3)
    cov4=np.cov(mean_vect4)
    covmats.append(cov4)
    cov5=np.cov(mean_vect5)
    covmats.append(cov5)
    covmats=np.array(covmats)
    
    # IMPLEMENT THIS METHOD
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    means,covmat=ldaLearn(X,y)
    mean = np.transpose(means)
    
    #mean = np.transpose(means)
    count=0
    ypred=[]
    total=len(ytest)
    
    n= Xtest.shape
    a=multivariate_normal.pdf(Xtest, mean[0], covmat)
    b=multivariate_normal.pdf(Xtest, mean[1], covmat)
    c=multivariate_normal.pdf(Xtest, mean[2], covmat)
    d=multivariate_normal.pdf(Xtest, mean[3], covmat)
    e=multivariate_normal.pdf(Xtest, mean[4], covmat)
    
    for i in range(len(Xtest)):
        if(max(a[i],b[i],c[i],d[i],e[i])==a[i]):
            ypred.append(1.0)
        if(max(a[i],b[i],c[i],d[i],e[i])==b[i]):
            ypred.append(2.0)
        if(max(a[i],b[i],c[i],d[i],e[i])==c[i]):
            ypred.append(3.0)
        if(max(a[i],b[i],c[i],d[i],e[i])==d[i]):
            ypred.append(4.0)
        if(max(a[i],b[i],c[i],d[i],e[i])==e[i]):
            ypred.append(5.0)
    for i in range(len(ytest)):
        if(ypred[i]==ytest[i]):
            count=count+1
    acc=count/total
    ypred=np.array(ypred)
    ypred=ypred.ravel()
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    means,covmats=qdaLearn(X,y)
    mean = np.transpose(means)
    count=-1
    ypred=[]
    total=len(ytest)
    
    cov1=covmats[0]
    cov2=covmats[1]
    cov3=covmats[2]
    cov4=covmats[3]
    cov5=covmats[4]
    
    n=Xtest.shape
    a=multivariate_normal.pdf(Xtest, mean[0], cov1)
    b=multivariate_normal.pdf(Xtest, mean[1], cov2)
    c=multivariate_normal.pdf(Xtest, mean[2], cov3)
    d=multivariate_normal.pdf(Xtest, mean[3], cov4)

    e=multivariate_normal.pdf(Xtest, mean[4], cov5)
    for i in range(len(Xtest)):
        if(max(a[i],b[i],c[i],d[i],e[i])==a[i]):
            ypred.append(1.0)
        if(max(a[i],b[i],c[i],d[i],e[i])==b[i]):
             ypred.append(2.0)
        if(max(a[i],b[i],c[i],d[i],e[i])==c[i]):
             ypred.append(3.0)
        if(max(a[i],b[i],c[i],d[i],e[i])==d[i]):
            ypred.append(4.0)
        if(max(a[i],b[i],c[i],d[i],e[i])==e[i]):
            ypred.append(5.0)
    for i in range(len(ytest)):
        if(ypred[i]==ytest[i]):
            count=count+1
    acc=(count/total)
    
    ypred=np.array(ypred)
    ypred=ypred.ravel()
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD  
    #X_trans=X.transpose
    
    w = np.matmul(np.matmul(inv(np.matmul(X.transpose(),X)),X.transpose()),y) 
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    
    res = np.matmul((np.subtract(ytest,np.matmul(Xtest,w))).transpose(),(np.subtract(ytest,np.matmul(Xtest,w))))
    mse=res/Xtest.shape[0]
    return mse

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    
    res = np.matmul((np.subtract(ytest,np.matmul(Xtest,w))).transpose(),(np.subtract(ytest,np.matmul(Xtest,w))))
    mse=res/Xtest.shape[0]
    return mse

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD      
   # wMAP = inv(λID + X(transpose)X)(Xtranspose)y
  
    
    
    
    #d(J(w))
    
    # IMPLEMENT THIS METHOD    
    mat=np.shape(X)
    
    w=np.matmul(np.matmul(inv(np.add(np.multiply(lambd,np.identity(mat[1])),np.matmul(X.transpose(),X))),X.transpose()),y)
    return w 
    
def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    
    
    w=w.reshape(w.size,1)
    #print(w1)
    #print(w1.shape)
    res=y-np.matmul(X,w)
    
    error= np.multiply(0.5,(np.add((np.matmul(res.transpose(),res)),np.multiply(lambd,np.matmul(w.transpose(),w)))))
    error_grad= np.subtract(np.multiply(lambd,w),np.matmul(X.transpose(),np.subtract(y,np.matmul(X,w))))
    error_grad=np.squeeze(np.array(error_grad))
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    
    arr=np.shape(x)
   
    Xp=np.zeros((arr[0],p+1))
    for i in range(arr[0]):
        Xp[i][0]=int(1)
    for m in range(arr[0]):
        for n in range(1,p+1):
            Xp[m][n]=((x[m])**n)
            
    return Xp



# Problem 1
# load the sample data                                                                 

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)

print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()
ytest=ytest.ravel()
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')
plt.savefig('Lda_Qda.png')
plt.show()


# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')


# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
#print(w)
#print(Xtest.shape)
w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)


print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))


# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.savefig('Ridge_train.png')
plt.subplot(1, 2, 2)

plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.savefig('Ridge_test.png')
plt.show()


# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)

i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.savefig('Gradient_ridge.png')
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses3)] # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.savefig('Non_linear_lopt')
plt.show()

