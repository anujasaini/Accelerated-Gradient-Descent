#ifndef _fista_h
#define _fista_h

#include "functor.h"
using namespace std;


class Fista {

private: 
	// hyperparameters
	float learning_rate;
    float momentum_coeff;
	bool stopping_criteria; // true for tolerance_limit and false for number of iterations
	int max_iter;
	float tolerance_limit;
	int iterations; 

public:

	Fista();
	
	Fista (float learning_rate,float momentum_coeff, float tolerance_limit, int max_iter);
	
	Fista (float learning_rate,float momentum_coeff, int iterations);

	float* solve(CustomFunctor* f);

	//float* maximize(CustomFunctor* f);

	void print_algo();

};

#endif
