
#ifndef _gd_h
#define _gd_h

#include "functor.h"


class GD {

private: 
	// hyperparameters
	float learning_rate;
	bool stopping_criteria; // true for tolerance_limit and false for number of iterations
	int max_iter;
	float tolerance_limit;
	int iterations; 

public:

	GD();
	
	GD(float learning_rate, float tolerance_limit, int max_iter);
	
	GD(float learning_rate, int iterations);

	float* solve(CustomFunctor* f);

	void print_algo();

};


#endif