# Accelerated-Gradient-Descent-
Exploring Optimization methods with parallel computation techniques

## Problem Statement 
Given a system of linear equations, solve it using the iterative optimization techniques, e.g. Gradient Descent and its faster variants. A system of linear equations is given by 
> ![first image](https://latex.codecogs.com/gif.latex?%5Ctextbf%7BAx%7D%20%3D%20%5Ctextbf%7Bb%7D) where A is the coefficient matrix and x is vector of variables. 


This problem can be converted to an optimization problem.
> ![](https://latex.codecogs.com/gif.latex?%5Ctextit%7Bmin%20F%28x%29%7D%20%3D%20%28Ax-b%29%5ET%28Ax-b%29)

Gradient of the above function is given by
> ![](https://latex.codecogs.com/gif.latex?%5Ctextit%7Bgrad%20F%28x%29%7D%20%3D%202A%5ET%28Ax-b%29)

The gradient at consecutive steps can grow large. To avoid this the original equation is divided by the max element in the equation or by the determinant of the  matrix A on both side.
