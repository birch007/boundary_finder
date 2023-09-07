'''Module containing boundary search class and boundary search methods'''
'''Copyright 2023 Alexey Kovalev

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import numpy as np
def none_bf_method(data):
  '''implements none boundary find method '''
  return False,np.array([])

def weighted_bf_method(data):
  '''Implements weighted boundary find method.
  Parameters:
  data (array): an array in hyperspace with bunary labels
  Returns: the weighted mean for the means of  differen labels'''

  #separate postive and negative values and find stdantad deviations
  data_positive=(data[data[:,-1]==1][:,:-1])
  std_positive=np.std(data_positive,axis=0)
  mean_positive=np.mean(data_positive,axis=0)

  data_negative=(data[data[:,-1]==-1][:,:-1])
  std_negative=np.std(data_negative,axis=0)
  mean_negative=np.mean(data_negative,axis=0)

  #difference of mean values for positive and negative labels
  d=mean_positive-mean_negative

  if sum(std_negative+std_positive)==0:
      q=0.5
  else: q=std_negative/(std_negative+std_positive)

  #finding weighted average bewteen the means
  coord=mean_negative+d*q

  return(True,coord)

class BoundaryFindMethod:
  '''Class for funding a boundary point for binary labeled data'''
  def __init__(self,method='none',ppdim=5):
    '''
    Initialisation
    Paramaters:
      method(str): a name for the method to find the boundary
      ppdim(int): points per dimension. Method is used if number of points is less or equal ppdim**dimension.
      '''
    
    if method== 'none':
          self.bf_method=none_bf_method
    if method== 'weighted':
          self.bf_method=weighted_bf_method

    self.ppdim=ppdim

  def find_coord(self,data):
    ''' finding the coordinate for boundary point'''
    #dond do anithing for 'none' method
    shape=data.shape
    N=shape[0]
    #cut-off number of points per sub-cell
    Nlim=self.ppdim**(shape[1]-1)
    #check if there are too many points in the sub-cell
    if shape[0]>Nlim:
      return False,np.array([])

    return self.bf_method(data)




class BoundaryFinder:

  '''
  class to use boundary finding procedure
  '''
  def __init__(self,x,y,xmin=[],xmax=[],purity=1.0,rnd_split=0.2,relabel=True,method='none',points_per_dim=5.0,repeat=1):
    '''Initialization function for the class BoundaryFinder
    Parameters:
    x(array): coordinates of data points
    y(array): binary labels
    xmin(array): minimum box coordinates
    xmax (array) maximum box coordinates
    purity(number): a number to define the purity of data set: 1 is copletly pure, 0 is 50% mixture
    rnd_split(number): random number to define the box split, 0<=rnd_number<=0.25
    relable(boolean): whete to relabel the data to (-1,1)
    method(string): method to determine the boundary point in impure dataset
    points_per_dim (int): numper of points per dimension to use the boundary finding method in a sub-cell
    repeat(int): number of repeats. If rnd_split  is not zero, can be used to generate more boundary points.
    '''
    self.coord=[]
    self.bf_method=BoundaryFindMethod(method,points_per_dim)
    self.repeat=repeat
    self.purity=purity
    #find global boundaries
    if xmin==[]:
        self.xmin=np.min(x,axis=0)
    else:  self.xmin=xmin
    if xmax==[]:
        self.xmax=np.max(x,axis=0)
    else:  self.xmax=xmax

    #set random split parameter
    if rnd_split==None:
        self.rnd_split=0.2
    else:  self.rnd_split=rnd_split

    #change labels to -1, 1
    if relabel:
        labels=np.unique(y)
        if len(labels)!=2: raise Exception(" Non-binary target")
        y[y==labels[0]]=-1
        y[y==labels[1]]=1
    #expand dimension  for labels
    y=np.expand_dims(y,axis=1)
    #expand dimensions for 1D case
    if len(x.shape)==1:
        x=np.expand_dims(x,1)

    self.dims=x.shape[1]
    #create data array
    self.data=np.hstack((x,y))

    self.ppdim=points_per_dim

  def addCoord(self,coord):
      '''Adds new coordinate point(s)
      Parameters:
      coord(array): new coordinate point(s)'''
      if len(self.coord)==0:
                            self.coord=coord
      else:
                            self.coord=np.vstack((self.coord,coord))

  def buildSubArray(self,x1,x2,i,data):
      '''Reduces data to data points in sub-cell in one dimension
      Parameters:
        x1(array) : minimum limit
        x2(array) maximum limit
        i(int): dimension index
        data(arra): original data
      Returns:
        reduced data
      '''
      xm=(x1+x2)/2
      xr=(x2-x1)/2
      N=data.shape[0]
      dim=data.shape[1]
      dd=np.array([]*dim)
      dd=data[abs(data[:,i]-xm)<=xr]
      return dd

  def splits(self,xmin,xmax):
      '''Define splitting position in a cell
      Parameters:
        xmin(array): minimul limts of the cell
        xmax(array): maximum limits for the cell
      Returns:
        splitting coordinates(array)
              '''
        #setting up an array of random numbers around 0.5
      N=self.dims
      split=np.random.uniform(low=0.5-self.rnd_split,high=0.5+self.rnd_split,size=N)

      #split along the first dimension
      xx=np.array([xmin[0],+xmin[0]+split[0]*(xmax[0]-xmin[0]),xmax[0]])
      #build an array "xx" of splitting points along all dimensions
      for i in range(N-1):
          nx=[xmin[i+1],+xmin[i+1]+split[i+1]*(xmax[i+1]-xmin[i+1]),xmax[i+1]]
          xx=np.vstack((xx,nx))
      return xx

  def subcell_minmax(self,xx,k):
      '''Find minimum and maximum position for the giving splits and sub-cell index
      Parameters:
        xx(array): splitting coordinates
        k(int): sub-cell index
      Returns:
        xxmin,xxmx: minimum and maximum positions for the sub-cell
        '''
      N=self.dims


      xxmax=[0]*N
      xxmin=[0]*N

      #find sub-array in the sub-cell
      for nn in range(N):

              xxmin[nn]=xx[nn,k[nn]]
              xxmax[nn]=xx[nn][k[nn]+1]
      return xxmin,xxmax


  def checkNd(self,xmin,xmax,data,coord):
        '''Recursive function to find boundary coordinates
        Parameters:
          xmin(array): minimum limits for the cell
          xmax (array):maximum limits for the cell
          data(array): data
          coord(array): found boundary points
        Returns:
          self.coord(array): update array with boundary points

        '''
        #array to control, the stop condition
        #N is the dimension of data space
        N=self.dims
        #list for indices of "+1" and "-1" calsses
        class_positive=[]
        class_negative=[]
        #sub-cells indices
        sub_cells=np.zeros((2**N,N),dtype=int)
        #bit array to identify each sub-cell
        bits=np.array([i for i in range(N)])


        xx=self.splits(xmin,xmax)


    #check if there are different species in sub-areas
        for idx in range(2**N):
            #    #convert to int indices
            for m in range(N):
                sub_cells[idx,m]=(idx//(2**m))%2
            k=sub_cells[idx,:]

            #minum and maxim for the sub-cell
            xxmin,xxmax=self.subcell_minmax(xx,k)

            d2=data
            #find sub-array in the sub-cell
            for nn in range(N):
                     d2=self.buildSubArray(xxmin[nn],xxmax[nn],nn,d2)

                    #check purity
            difference=int(sum(d2[:,N]))
            if abs(difference)>=self.purity*len((d2[:,N])):

                if difference>0: class_positive.append(idx)
                if difference<0: class_negative.append(idx)

            else:
                result,new_coord=self.bf_method.find_coord(d2)
                if result:
                  self.addCoord(new_coord)
                else:
                 self.coord=self.checkNd(xxmin,xxmax,d2,self.coord)

        #find neigbors from different classes
        for idx_plus in class_positive:
            for idx_minus in class_negative:

                dist=abs(sub_cells[idx_plus,:]-sub_cells[idx_minus,:])

                if sum(dist)==1:
                        bit=sum(bits*dist)
                        #get minimum index of two neighbors
                        idx_base=min(idx_plus,idx_minus)
                        #conver index to array of bits
                        idx_k=sub_cells[idx_base,:]
                        new_coord=np.array([0.0]*N)
                        for i in range(N):
                            new_coord[i]=(xx[i,idx_k[i]]+xx[i,idx_k[i]+1])/2
                        new_coord[bit]=xx[bit,1]

                        if len(self.coord)==0:
                            self.coord=new_coord
                        else:
                            self.coord=np.vstack((self.coord,new_coord))
                        #self.append(data)

        return self.coord

  def fit(self):
        '''Fitting function'''
        #repeants self.repeat times
        for i in range(self.repeat):
          self.checkNd(self.xmin,self.xmax,self.data,self.coord)

