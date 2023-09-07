'''Helper functions'''
def make_dataND(fun,w,xmin,xmax,N=100):

  ''' Generate binary classes based on some boundary function in a box.
  Parameters:
    fun (function) : a function name
    w (any type): parameters for the function

    xmin, xmax: low and high bounds for unifrmly distributed points
    N(int) the number of points
    Returns:
        data(array): first columns are coordinates, last column  contains labels.
    '''
  dim=len(xmin)
  #create the (N, dim) array of randomly distributed points
  data=np.random.uniform(low=xmin,high=xmax,size=(N,dim))
  #calculate function and assign labels
  y=fun(data,w)>0
  #add labels to the data array
  y=(2*np.transpose([y])-1).astype('int')
  data=np.hstack((data,y))

  return data
  
def uni_noise(xmin,xmax,N=100):
    '''Creates random uniform noise data
    Parameters:
        xmin, xmax: low and high bounds for unifrmly distributed points
        N(int) the number of points
    Returns:
        data(array): first columns are coordinates, last column  contains labels.
    '''
    
    dim=len(xmin)
  #create the (N, dim) array of randomly distributed points
    data=np.random.uniform(low=xmin,high=xmax,size=(N,dim))
    y=np.random.uniform(size=N)>=0.5
    y=(2*np.transpose([y])-1).astype('int')
    data=np.hstack((data,y))
    return data
  
def sphere(x,w):
  ''' Calculates labels for the sphere
  Parameters:
   x(number): data in the hyperspace
   r: sphere radius
   Returns: sdifference of the squared distance to the origin
   and the squared radius
   '''
  x=np.array(x)*np.array(w[1:])
  return (sum(x*x)-w[0]**2)

def dersphere(x,w):
  '''
  Reurn the gradient for the sphere
  '''
  x=np.array(x)
  w=np.array(w[1:])
  w=w*w
  return 2*x*w

def plane(x,w):
  return sum(x*w[1:]+w[0])
def derplane(x,w):
  return w[1:]

def func(x,w,fun,derfun,point):
    grad=derfun(x[:-1],w)
    k=len(w)
    r=[fun(x[:-1],w)]
    for i in range(k-1):
        r.insert(i,grad[i]*x[k-1]+x[i]-point[i])
    return r

import numpy as np
from scipy.optimize import fsolve

def find_dist(fun,derfun,point,w):
    x0=point.copy()
    x0.append(0)
    root=fsolve(func, x0,args=(w,fun,derfun,point))
    d=np.array(root[:-1])-np.array(point)
    return np.sqrt(sum(d*d))
def fun2(x,w,fun):
  '''
  creates an array from single-valued function
  Parameters:
  x(number): array of points in hyperspace
  w(any type): parameter for the function fun
  fun(function) : a sigle valued function
  '''
  res=[]
  N=x.shape[0]
  for i in range(N):
      res.append(fun(x[i],w))
  return np.array(res)


def sphere2(x,w):
  return fun2(x,w,sphere)