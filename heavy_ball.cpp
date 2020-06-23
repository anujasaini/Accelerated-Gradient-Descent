#include "heavy_ball.h"
#include "utils.h"
HeavyBall::HeavyBall(){
	this->learning_rate = 0.001;
	this->momentum_coeff = 0.1;
	this->stopping_criteria = false;
	this->max_iter = 10000;
	this->iterations = 1000;
}

HeavyBall::HeavyBall(float learning_rate,float momentum_coeff, float tolerance_limit, int max_iter){
	this->learning_rate = learning_rate;
	this->momentum_coeff = momentum_coeff;
	this->stopping_criteria = true;
	this->max_iter = max_iter;
	this->tolerance_limit = tolerance_limit;
}

HeavyBall::HeavyBall(float learning_rate,float momentum_coeff, int iterations){
	this->learning_rate = learning_rate;
	this->momentum_coeff = momentum_coeff;
	this->stopping_criteria = false;
	this->iterations = iterations;
	this->tolerance_limit = 1e-10;

}


float* HeavyBall::solve(CustomFunctor* f){

	float* x = new float[f->n]{0};

	if(stopping_criteria){
		// tolerance limit
		float error = INT_MAX;
		int iter_steps = 0;
		while(error >= this->tolerance_limit and iter_steps < this->max_iter){
			iter_steps++;
			float prev_val = f->eval(x);
			x = sub(x, scalar_mult(f->grad(x), this->learning_rate, f->n), f->n, f->n);
			float next_val = f->eval(x);
			// cout<<"Value of function after "<<iter_steps<<" : "<<next_val<<endl;
			error = abs(prev_val - next_val);
		}

		if(iter_steps != max_iter){
			cout<<"Gradient decent converges after :"<<iter_steps<<endl; 
		}else{
			cout<<"Method didn't converge!!!";
		}

	}else{
		int iter_steps = 0;
		float error = INT_MAX;

        float* prev = new float[f->n];
			float * temp = new float[f->n];

            for(int i=0;i<f->n;i++)
				{
                    prev[i]= x[i];
                    temp[i] = x[i];
                }

        x = add(sub(temp,scalar_mult(f->grad(temp), this->learning_rate, f->n), f->n, f->n), scalar_mult(sub(temp, prev, f->n, f->n), this->momentum_coeff, f->n), f->n, f->n);

		for(int i=0;i<this->iterations;i++){
			iter_steps++;
			for(int i=0;i<f->n;i++)
				{
                    temp[i] = x[i];
                }
			x = add(sub(temp,scalar_mult(f->grad(temp), this->learning_rate, f->n), f->n, f->n), scalar_mult(sub(temp, prev, f->n, f->n), this->momentum_coeff, f->n), f->n, f->n);
			// cout<<"Value of function after "<<iter_steps<<" : "<<f->eval(x)<<" "<<endl;

			for(int i=0;i<f->n;i++)
				{
                    prev[i]= temp[i];
                }

			error = 0.0;
			for(int i=0;i<f->n;i++) error += (temp[i]-x[i])*(temp[i]-x[i]);
			if(error < this->tolerance_limit ) break;
		}

			if(iter_steps != iterations){
				cout<<"Gradient decent converges after :"<<iter_steps<<endl; 
			}else{
				cout<<"Method didn't converge!!!";
			}

	}

	return x;

}
	

void HeavyBall::print_algo(){

	string algo = "Gradient Decent";
	cout<<algo;
}
