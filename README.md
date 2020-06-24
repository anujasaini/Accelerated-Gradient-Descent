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

so hopefully the sequence ![](https://latex.codecogs.com/gif.latex?%28%5Ctextbf%7B%24x%20_%7Bn%7D%24%7D%29) converges to the desired local minimum. Note that the value of the step size ![](https://latex.codecogs.com/gif.latex?%5Cgamma) is allowed to change at every iteration. With certain assumptions on the function F (for example, F convex and ![](https://latex.codecogs.com/gif.latex?%5Cnabla%20F) Lipschitz) and particular choices of ![](https://latex.codecogs.com/gif.latex?%5Cgamma), 
convergence to a local minimum can be guaranteed. When the function F is convex, all local minima are also global minima, so in this case gradient descent can converge to the global solution.

### Complexity and Convergence Analysis
#### Complexity

- For strongly convex and smooth functions, the convergence rate of gradient descent is ![](https://latex.codecogs.com/gif.latex?O%28klog%28%5Cfrac%7B1%7D%7B%5Cepsilon%7D%29%29), where ![](https://latex.codecogs.com/gif.latex?%5Cepsilon) is error tolerance.
- For convex and smooth functions, the convergence rate of gradient descent is ![](https://latex.codecogs.com/gif.latex?O%28%5Cfrac%7B1%7D%7B%5Cepsilon%7D%29)

#### Convergence Analysis
- Gradient Descent algorithm focuses on improving cost per iteration, which might be too “short-sighted” view. So, we can exploit information from past iterations to get better idea of gradient.
- Gradient Descent algorithm might sometimes zigzag or experience abrupt changes. So we can add buffer (like momentum) to yield smoother trajectory and hence avoid zigzag or abrupt changes.


## Heavy Ball Method 
### Introduction 

We consider the objective function ![](https://latex.codecogs.com/gif.latex?f) to be l-strongly convex and L-smooth. Thus,
![](https://latex.codecogs.com/gif.latex?lI%20%5Cleq%20%5Cnabla%5E%7B2%7Df%20%5Cleq%20LI%20%24%2C%20where%20%24%5Ckappa%20%3D%20%5Cfrac%7BL%7D%7Bl%7D). 
![](https://latex.codecogs.com/gif.latex?I%20%3D%20Identity%20Matrix) 

The heavy-ball method is a multi-step iterative method that exploits iterates prior to the most recent one. It does so by maintaining the momentum of the previous two iterates. If you imagine each iterate as a point in the trajectory of a falling ball, then the points carry with them a momentum or a velocity. The heavy ball method is one way of guarding against zigzagging trajectories in iterative descent.

### Algorithm 

Our task is to minimize the function ![](https://latex.codecogs.com/gif.latex?f%28%5Ctextbf%7Bx%7D%29) where ![](https://latex.codecogs.com/gif.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D). Heavy ball method is also called chebyshev
iterative method. Its update rule is
> ![](https://latex.codecogs.com/gif.latex?x%5E%7Bk&plus;1%7D%20%3D%20x%5E%7Bk%7D%20-%20%5Cgamma%20%5Chspace%7B1mm%7D%20%5Cnabla%20f%28x%29%20&plus;%20%5Cbeta%20%28x%5E%7Bk%7D%20-%20x%5E%7Bk-1%7D%29)

### Complexity and Convergence Analysis
The convergence rate of heavy ball method is ![](https://latex.codecogs.com/gif.latex?O%28%5Csqrt%7B%5Ckappa%7D%20log%28%5Cfrac%7B1%7D%7B%5Cepsilon%7D%29%29)

## Nesterov's accelerated gradient methods

### Introduction 
Nesterov's Accelerated Gradient Descent performs a simple step of gradient descent to go from ![](https://latex.codecogs.com/gif.latex?%7Bx_%7Bs%7D%7D%24%20to%20%24%7By_%7Bs&plus;1%7D%7D), and then it ‘slides’ a little bit further than ![](https://latex.codecogs.com/gif.latex?%7By_%7Bs&plus;1%7D%7D) in the direction given by the previous point ![](https://latex.codecogs.com/gif.latex?%7By_%7Bs%7D%7D.)

It alternates between gradient updates and proper extrapolation.   each iteration takes nearly same cost as GD. It is not a descent method (i.e. we may not have ![](https://latex.codecogs.com/gif.latex?f%28x%5E%7Bk%7D%20&plus;%201%29) ![](https://latex.codecogs.com/gif.latex?%5Cleq%20f%28x%5E%7Bk%7D%29%29)

### Algorithm 

In order to minimize the function ![](https://latex.codecogs.com/gif.latex?f%28%5Ctextbf%7Bx%7D%29%24%20where%20x%20%24%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D). The update rule is:
> ![](https://latex.codecogs.com/gif.latex?x%5E%7Bk&plus;1%7D%20%3D%20y%5E%7Bk%7D%20-%20%5Cgamma%20%5Chspace%7B1mm%7D%20%5Cnabla%20f%28y%5E%7Bk%7D%29)
> ![](https://latex.codecogs.com/gif.latex?y%5E%7Bk&plus;1%7D%20%3D%20x%5E%7Bk&plus;1%7D%20&plus;%20%5Cfrac%7Bk%7D%7Bk&plus;3%7D%20%28x%5E%7Bk&plus;1%7D%20-%20x%5E%7Bk%7D%29)

### Complexity and Convergence Analysis

The iteration complexity of the algorithm is ![](https://latex.codecogs.com/gif.latex?O%28%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Cepsilon%7D%7D%29). This overall is much faster than the gradient decent. Also, the coefficient ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bk%7D%7Bk&plus;3%7D) can be shown to be mystically working using the differential equation and damped spring based motion.

## Accelerated proximal gradient methods (FISTA)

### Introduction

Fast iterative shrinkage-thresholding algorithm (FISTA) preserves the computational simplicity of FISTA but with a global rate of convergence which is proven to be significantly better, both theoretically and practically. It again is not a descent method but works as well as Nesterov.

### Algorithm 

In order to minimize the function ![](https://latex.codecogs.com/gif.latex?f%28%5Ctextbf%7Bx%7D%29) where ![](https://latex.codecogs.com/gif.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D). The update rule is:
> ![](https://latex.codecogs.com/gif.latex?x%5E%7Bk&plus;1%7D%20%3D%20prox_%7B%5Cgamma%20h%7D%28y%5E%7Bk%7D%20-%20%5Cgamma%20%5Chspace%7B1mm%7D%20%5Cnabla%20f%28y%5E%7Bk%7D%29)
> ![](https://latex.codecogs.com/gif.latex?y%5E%7Bk&plus;1%7D%20%3D%20x%5E%7Bk&plus;1%7D%20&plus;%20%5Cfrac%7B%5Ctheta_%7Bk%7D%20-%201%7D%7B%5Ctheta_%7Bk&plus;1%7D%7D%20%28x%5E%7Bk&plus;1%7D%20-%20x%5E%7Bk%7D%29)

where ![](https://latex.codecogs.com/gif.latex?y%5E%7B0%7D%3Dx%5E%7B0%7D), 
> ![](https://latex.codecogs.com/gif.latex?%5Ctheta_%7B0%7D%20%3D%201%20and%20%5Ctheta_%7Bk&plus;1%7D%20%3D%20%5Cfrac%7B1%20&plus;%20%5Csqrt%7B1%20&plus;%204%5Ctheta_%7Bk%7D%5E%7B2%7D%29%7D%7D%7B2%7D)

Here, the proximal function is taken to be zero. In general, proximal of any function with parameter ![](https://latex.codecogs.com/gif.latex?%5Cgamma) is given by
> ![](https://latex.codecogs.com/gif.latex?%5Ctextit%7B%24prox_%7B%5Cgamma%20f%7D%28v%29%24%7D%20%3D%20argmin%28f%28x%29%20&plus;%20%5Cgamma%20%7C%7Cx-v%7C%7C%5E%7B2%7D%29)
if the function is taken is taken as f(x) = constant, which is the case with us, proximal will be same as v. Hence, the update rule remains the same. 


### Complexity and Convergence Ananlysis 

FISTA has improved iteration complexity (i.e. ![](https://latex.codecogs.com/gif.latex?O%28%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Cepsilon%7D%7D%29)) than proximal
gradient method (i.e. ![](https://latex.codecogs.com/gif.latex?O%28%5Cfrac%7B1%7D%7B%5Cepsilon%7D%29)). Also, fast if proximal can be efficiently implemented. Convergence is similar to that of the Nesterov since coefficient ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Ctheta_%7Bk%7D%20-%201%7D%7B%5Ctheta_%7Bk&plus;1%7D%7D)  is asymptotically equivalent to ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bk%7D%7Bk&plus;3%7D)

## Results from Serial Code 
Convergence steps required for the  each algorithm vs the size of the problem i.e n is shown.
![](https://github.com/anujasaini/Accelerated-Gradient-Descent-/blob/master/img/steps.png = 250X250) ![](https://github.com/anujasaini/Accelerated-Gradient-Descent-/blob/master/img/time_taken.png) 
