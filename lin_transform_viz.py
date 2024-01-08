#%%
import numpy as np
%matplotlib inline
from matplotlib import pyplot

#%%
## Source : https://github.com/engineersCode/EngComp4_landlinear
# from urllib.request import urlretrieve
# URL = 'https://go.gwu.edu/engcomp4plot'  
# urlretrieve(URL, 'plot_helper.py')
#%%
from plot_helper import *

A = np.array([[3, 2],
              [-2, 1]])

plot_linear_transformation(A)

#%%
from plot_helper import *

A = np.array([[0, -1],
              [1, 0]])

plot_linear_transformation(A)

#%%

theta = np.pi / 4

A = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

plot_linear_transformation(A)

# %%
A = np.array([[1, 1],
              [1, 1]])

plot_linear_transformation(A)
# %%
A = np.array([[5, 1],
              [10, 2]])

plot_linear_transformation(A)
# %%
