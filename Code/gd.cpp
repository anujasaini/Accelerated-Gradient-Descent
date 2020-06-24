#include "gd.h"
#include "utils.h"
GD::GD(){
	this->learning_rate = 0.001;
	this->stopping_criteria = false;
	this->max_iter = 10000;
	this->iterations = 1000;
}

GD::GD(float learning_rate, float tolerance_limit, int max_iter){
	this->learning_rate = learning_rate;
	this->stopping_criteria = true;
	this->max_iter = max_iter;
	this->tolerance_limit = tolerance_limit;
}

GD::GD(float learning_rate, int iterations){
	this->learning_rate = learning_rate;
	this->stopping_criteria = false;
	this->iterations = iterations;
	this->tolerance_limit = 1e-10;

}


float* GD::solve(CustomFunctor* f){

	float* x = new float[f->n]{0};
	float* temp;
	if(stopping_criteria){
		// tolerance limit
		float error = INT_MAX;
		int iter_steps = 0;
		cout<<"dwdawd";
		temp = new float[f->n];
		while(error >= this->tolerance_limit and iter_steps < this->max_iter){
			iter_steps++;
			float prev_val = f->eval(x);

			memcpy(temp, x, f->n*sizeof(float));
			
			x = sub(temp, scalar_mult(f->grad(temp), this->learning_rate, f->n), f->n, f->n);
			
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

		// iterations 
		float error = INT_MAX;
		int iter_steps = 0;
		temp = new float[f->n];

		for(int i=0;i<this->iterations;i++){
			iter_steps++;
			// float* f1 = f->grad(x);

			memcpy(temp, x, f->n*sizeof(float));
			float prev_val = f->eval(x);

			x = sub(temp, scalar_mult(f->grad(temp), this->learning_rate, f->n), f->n, f->n);
			// cout<<"Value of function after "<<iter_steps<<" : "<<f->eval(x)<<endl;
			
			float next_val = f->eval(x);

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


void GD::print_algo(){

	string algo = "Gradient Decent";
	cout<<algo;
}
