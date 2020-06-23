

#ifndef _functor_h
#define _functor_h
 
#include "linear_eqns.h"


class CustomFunctor{

public:	
		float** A;
		float* b;
		int n;
		int m;

		CustomFunctor(float** A, float* b, int n, int m);

		CustomFunctor(Equation* e);

		float eval(float* x);

		float* grad(float* x);

};


#endif
