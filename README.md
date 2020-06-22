# Accelerated-Gradient-Descent-
Exploring Optimization methods with parallel computation techniques

Given a system of linear equations, solve it using the iterative optimization techniques, e.g. Gradient Descent and its faster variants. A system of linear equations is given by

\begin{center}
    \textbf{Ax} = \textbf{b}
\end{center}

where A is the coefficient matrix and  x is vector of variables. This problem can be converted to an optimization problem. 

\begin{center}
    \textit{min F(x)} $ = (Ax-b)^T(Ax-b)$
\end{center}

Gradient of the above function is given by

\begin{center}
    \textit{grad F(x)} $ = 2A^T(Ax-b)$
\end{center}

The gradient at consecutive steps can grow large. To avoid this the original equation is divided by the max element in the equation or by the determinant of the  matrix A on both side.
