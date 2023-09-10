# boundary_finder
Finding boundaries between two classes.
boundarysearch.py  contains classes  and methods for boundary finder object.
helpfunc.py contains helper functions to create synthetic datasets and to estimate the distance from a boundary point to an "ideal" boundary.
boun_finder.ipnb is an example notebook that shows how to use the boundary finder.

*class* **BoundaryFinder**(*x,y,xmin=[],xmax=[],purity=1.0,rnd_split=0.2,relabel=True,method='none',points_per_dim=5.0,repeat=1*)
**Parameters**: 
  x: array, features 
  y: array,labels
  xmin: list,minimum values for features
  xmax: list,maximum values for features
  purity: float, purity value to determine the single-label region
  rnd_split: float, 0-0.25, to define the random splitting of sub-cells
  relabel: bool, determines if the data need to be relabeled  to [-1,1]
  method: str. Possible values "none' or 'weighted'. Defines method to determine the boundary point
  points_per_dim: int, defines limiting number of datapoints per dimension for 'weighted' method
  repeat: int, defines how many times the algorithm runs. Makes sense only of rnd_split is not zero
  
  



