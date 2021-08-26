# CppLinAlg
MATLAB is known for having linear algebra capabilities such as matrix arithmetic and linear system solvers. This project contains a C++ impementation of Matrix and Vector classes, with MATLAB like functionality. Capabilities include solving linear systems using GMRES and CG solvers, a determinant finder and basic matrix arithmetic.

Matrix.cpp and Vector.cpp contain methods belonging the the matrix and vector classes and some supplementary linear algebra functions.

use_vectors.cpp contains a test function that can be used ensure existing functionality still works as new functionality is added. use_vectors.cpp also demonstrates how to use each of the functions in the Matrix.cpp and Vector.cpp classes.

plot_res.m is a MATLAB function that can be used to plot the residuals at each iteration of the linear solvers.
