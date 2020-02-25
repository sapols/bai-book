import numpy as np
import pandas as pd
from nltools.external import glover_hrf
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import sin,pi,arange
from scipy.ndimage import gaussian_filter1d
def compute_design_vector(d,l,onset_type_number,total_scan_time,tr,padding=0):
 X=np.floor(np.array(d)/(tr*1000)).astype(int)
 z=X[onset_type_number==np.array(l)]
 n=int(total_scan_time/tr)+padding
 v=np.zeros(n)
 v[z]=1
 return v
def generate_time_series(A,betas):
 y=np.zeros(A.shape[0]) 
 for I,beta in zip(A.columns,betas):
  y=y+A[I]*beta
 return y
def estimate_beta(X,Y):
 return np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),Y)
def compute_contrast_2d(N,contrast): 
 b=np.squeeze(N) 
 c=np.array(contrast).T
 cs=np.dot(b,c) 
 return cs
class FunctionalDataSim2D:
 @staticmethod
 def plot_beta_map(N,dm):
  plt.figure(figsize=(40,4))
  o=dm.columns
  m=len(o) 
  for i,w in enumerate(o):
   D=plt.subplot(m,1,i+1)
   plt.imshow(N[:,:,i],cmap='bwr',vmin=-1,vmax=1)
   plt.ylabel(w,fontsize=12,rotation=0,labelpad=40,va='center')
   D.get_yaxis().set_ticks([]) 
 @staticmethod
 def compute_beta_map(J,X):
  g=J.shape[0]
  B=J.shape[1]
  n=X.shape[1]
  N=np.zeros((g,B,n)) 
  for i in range(g):
   for j in range(B):
    Y=J[i,j,:]
    bs=estimate_beta(X,Y)
    N[i,j,:]=bs
  return N
 @staticmethod
 def smooth(J):
  J=gaussian_filter1d(J,sigma=1,axis=0,output=J)
  J=gaussian_filter1d(J,sigma=1,axis=1,output=J) 
 @staticmethod
 def create(g,B,W):
  J=np.zeros((g,B,W))
  return J
 @staticmethod
 def add(J,A,betas,locations):
  for i,j in locations:
   J[i,j,:]=generate_time_series(A,betas)
 @staticmethod
 def plot_data(J,selected=[]):
  plt.figure(figsize=(40,4))
  W=J.shape[2]
  E=selected
  m=len(E)
  for i,R in enumerate(E):
   D=plt.subplot(m,1,i+1)
   plt.imshow(J[:,:,R],vmin=-1,vmax=1,cmap='bwr')
   plt.ylabel(str(R),fontsize=20,rotation=0,labelpad=50,va='center')
   D.get_yaxis().set_ticks([])
 @staticmethod
 def show_contrast(N,contrast):
  cs=compute_contrast_2d(N,contrast)
  plt.figure(figsize=(40,2))
  plt.imshow(cs,cmap='gray')
class DesignMatrixUtils:
 @staticmethod
 def add_many(A,funcs=[]):
  r=A.copy()
  for f in funcs:
   r=f(r)
  return r
 @staticmethod
 def add_cosine(A,M,amplitude,phase=0): 
  c=A.shape[0]
  R=np.linspace(0,1,c)
  F=amplitude*sin(2*pi*M*R+pi*phase)
  V=A.copy()
  V['cos'+str(M)]=F
  return V
 @staticmethod
 def add_poly(A,degrees=2,flipped=False): 
  R=np.linspace(-1,1,A.shape[0])
  if flipped:
   s=1-R**degrees
   S='1-poly'+str(degrees)
  else:
   s=R**degrees
   S='poly'+str(degrees)
  V=A.copy()
  V[S]=s
  return V
 @staticmethod
 def convolve_hrf(A,tr):
  t=glover_hrf(tr,oversampling=1) 
  V=A.copy()
  for c in A.columns:
   V[c]=np.convolve(V[c].to_numpy(),t,mode='same') 
  return V
 @staticmethod
 def heatmap(A):
  o=A.columns
  sns.heatmap(A,xticklabels=o)
 @staticmethod
 def add_intercept(A):
  V=A.copy()
  V['intercept']=np.ones(A.shape[0])
  return V
 @staticmethod
 def create_design_matrix(events,o,tr,c):
  m=len(o)
  A=np.zeros((m,c),int)
  d=[onset_time for(onset_time,onset_type)in events]
  l=[onset_type for(onset_time,onset_type)in events]
  b=c*tr 
  for i in range(0,len(o)):
   U=i 
   x=compute_design_vector(d,l,U,b,tr,0)
   A[i,:]=x 
  A
  df=pd.DataFrame(data=A.T,columns=o)
  return df
 @staticmethod
 def plot(A,orientation='horizontal'):
  o=A.columns 
  c,m=A.shape 
  E=range(c)
  if orientation=='horizontal':
   f,a=plt.subplots(nrows=m,figsize=(10,m),sharey=True)
   for i,w in enumerate(o): 
    a[i].plot(E,A[w].to_numpy()) 
    a[i].set_title(w,fontsize=12,rotation=0)
  else:
   f,a=plt.subplots(ncols=m,figsize=(m*1.5,10),sharey=True)
   for i,w in enumerate(o): 
    a[i].plot(A[w].to_numpy(),E) 
    a[i].set_title(w,fontsize=12,rotation=45)
   plt.gca().invert_yaxis() 
  plt.tight_layout()
# Created by pyminifier (https://github.com/liftoff/pyminifier)

