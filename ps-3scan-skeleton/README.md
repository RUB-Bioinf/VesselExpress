# 3D Image Skeletonization Tools

This repository contains programs needed to obtain a 3D skeleton and quantify the skeletonized array to statistics with the help of function present in metrics.segmentStats.  
Input must be a binary array with z in its first dimension.  
Skeletonization on a 3D binary array is performed by iteratively removing the boundary points until a line in the center is obtained by convolving the image with structuring elements from the [paper](https://drive.google.com/file/d/1kCEmfOx1mwoyggAfkYOsyRhOywfkiIU1/view?usp=sharing).  
This function is implemented using [cython](http://docs.cython.org/src/reference/compilation.html) for fast execution and pyximport is used to automatically build and use the function

Using mayavi is a useful way to visualize these test stacks  
Install mayavi via:  

```conda install -c menpo mayavi=4.5.0```

This version works on python 3  
Versions < 4.5 don't work on python 3.x  
mayavi also requires that QT4 (pyqt) is installed  
This may be in conflict with matplotlib >=1.5.3, which started to use qt5. So, use mayavi and matplotlib in a separate [conda](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04) environment to avoid conflicts

View the stack contours:

```import mayavi as mlab```  
```mlab.contour3d(anynpynonbooleanarray)```  
```mlab.options.offscreen = True```  
```mlab.savefig("arrayName.png")```  
