
#ifndef _linear_equation_h
#define _linear_equation_h

#include <bits/stdc++.h>
using namespace std;
class Equation {

public:

	float** A;
	float* b;
	int n;
	int m;


	// n equations in m variables with random coefficients
	Equation(int n, int m);

	// m equations in n variables with coeficients as given by 2D array
	Equation(float** coeffs, float* b,  int n, int m);

	// will evaluate the equations on values provided
	float* evaluate(float* values);

	// print the equations 
	void stringify();
};


#endif