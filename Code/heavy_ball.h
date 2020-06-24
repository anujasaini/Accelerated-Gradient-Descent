#ifndef _heavy_ball_h
#define _heavy_ball_h


#include "functor.h"
using namespace std;


class HeavyBall {

private: 
	// hyperparameters
	float learning_rate;
    float momentum_coeff;
	bool stopping_criteria; // true for tolerance_limit and false for number of iterations
	int max_iter;
	float tolerance_limit;
	int iterations; 

public:

	HeavyBall();
	
	HeavyBall (float learning_rate,float momentum_coeff, float tolerance_limit, int max_iter);
	
	HeavyBall (float learning_rate,float momentum_coeff, int iterations);

	float* solve(CustomFunctor* f);

	void print_algo();

};


#endif