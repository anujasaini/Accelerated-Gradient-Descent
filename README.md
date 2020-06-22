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

## Gradient Descent

### Introduction 
Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point.

### Algorithm 
Gradient descent is based on the observation that if the multi-variable function F(\textbf{x})  is defined and differentiable in a neighborhood of a point ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%20%7Ba%7D%2C%20then%20F%28%5Ctextbf%7Bx%7D%29) decreases fastest if one goes from ![](https://latex.codecogs.com/gif.latex?F%28%5Ctextbf%7Ba%7D%29) in the direction of the negative gradient of F at ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7Ba%7D%2C%20i.e.%2C%20-%5Cnabla%20F%28%5Ctextbf%7Ba%7D%29).  
It follows that, if
> ![](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Ba_%7Bn&plus;1%7D%20%3D%20a_%7Bn%7D%20-%20%5Cgamma%20%5Cnabla%20F%28a_%7Bn%7D%29%7D)


for ![](https://latex.codecogs.com/gif.latex?%5Cgamma%20%5Cin%20%5Cmathbb%7BR_%7B&plus;%7D%7D) small enough, 
then ![](https://latex.codecogs.com/gif.latex?F%28%5Cmathbf%7Ba%7D%29%20%5Cgeq%20F%28%5Cmathbf%20%7Ba_%7Bn&plus;1%7D%7D%29). 
In other words, the term ![](https://latex.codecogs.com/gif.latex?%5Cgamma%20%5Cnabla%20F%28%5Ctextbf%7Ba%7D%29) is subtracted from ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7Ba%7D) because we want to move against the gradient, towards the minimum. 

With this observation in mind, one starts with a guess ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7B%24x_%7B0%7D%24%7D) for a local minimum of F, and considers the sequence ![](https://latex.codecogs.com/gif.latex?%5Ctextbf%7B%24x_%7B0%7D%24%7D%2C%20%5Ctextbf%7B%24x_%7B1%7D%24%7D%2C%20%5Ctextbf%7B%24x_%7B2%7D%24%7D%2C%20%5Cdots), such that

> ![](https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%20_%7Bn&plus;1%7D%7D%20%3D%20%5Cmathbf%7Bx%20_%7Bn%7D%7D%20-%20%5Cgamma%20_%7Bn%7D%20%5Cnabla%20F%28%5Cmathbf%20%7Bx%20_%7Bn%7D%7D%29%2C%5C%20n%5Cgeq%200.)


We have a monotonic sequence
> ![](https://latex.codecogs.com/gif.latex?F%28%5Ctextbf%7B%24%20x_%7B0%7D%20%24%7D%29%20%5Cgeq%20%24%20F%28%5Ctextbf%7B%24%20x_%7B1%7D%20%24%7D%29%20%24%20%5Cgeq%20%24%20F%28%5Ctextbf%7B%24%20x%20_%7B2%7D%20%24%7D%29%20%24%5Cgeq%20%5Ccdots),

so hopefully the sequence ![](https://latex.codecogs.com/gif.latex?%28%5Ctextbf%7B%24x%20_%7Bn%7D%24%7D%29) converges to the desired local minimum. Note that the value of the step size ![](https://latex.codecogs.com/gif.latex?%5Cgamma) is allowed to change at every iteration. With certain assumptions on the function F (for example, F convex and $\nabla$F Lipschitz) and particular choices of ![](https://latex.codecogs.com/gif.latex?%5Cgamma), 
convergence to a local minimum can be guaranteed. When the function F is convex, all local minima are also global minima, so in this case gradient descent can converge to the global solution.

