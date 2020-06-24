#include "nesterov.h"
#include "utils.h"
Nesterov::Nesterov(){
	this->learning_rate = 0.001;
	this->momentum_coeff = 1;
	this->stopping_criteria = false;
	this->max_iter = 10000;
	this->iterations = 1000;
}

Nesterov::Nesterov(float learning_rate,float momentum_coeff, float tolerance_limit, int max_iter){
	this->learning_rate = learning_rate;
	this->momentum_coeff = momentum_coeff;
	this->stopping_criteria = true;
	this->max_iter = max_iter;
	this->tolerance_limit = tolerance_limit;
}

Nesterov::Nesterov(float learning_rate,float momentum_coeff, int iterations){
	this->learning_rate = learning_rate;
	this->momentum_coeff = momentum_coeff;
	this->stopping_criteria = false;
	this->iterations = iterations;
	this->tolerance_limit = 1e-10;

}


float* Nesterov::solve(CustomFunctor* f){

	float* x = new float[f->n]{0};
    float* y = new float[f->n]{0};
    float* prev = new float[f->n]{0};
	if(stopping_criteria){
		// tolerance limit
		float error = INT_MAX;
		int iter_steps = 0;
        int t=0;
		while(error >= this->tolerance_limit and iter_steps < this->max_iter){
			iter_steps++;
			float prev_val = f->eval(x);
            memcpy(prev, x, f->n*sizeof(float));

			x = sub(y, scalar_mult(f->grad(y), this->learning_rate, f->n), f->n, f->n);
            float g = (t/(t+3)); t++;
            y = add(x, scalar_mult(sub(x, prev, f->n, f->n),g, f->n), f->n, f->n);


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
        int t=0;
           
        memcpy(prev, x, f->n*sizeof(float));
        x = sub(y, scalar_mult(f->grad(y), this->learning_rate, f->n), f->n, f->n);
        float g = (t/(t+3)); t++;
        y = add(x, scalar_mult(sub(x, prev, f->n, f->n),g, f->n), f->n, f->n);

		for(int i=0;i<this->iterations;i++){
			iter_steps++;
    		float error = INT_MAX;


			memcpy(prev, x, f->n*sizeof(float));
            x = sub(y, scalar_mult(f->grad(y), this->learning_rate, f->n), f->n, f->n);
            float g = (t/(t+3)); t++;
            y = add(x, scalar_mult(sub(x, prev, f->n, f->n),g, f->n), f->n, f->n);

            // cout<<"Value of function after "<<iter_steps<<" : "<<f->eval(x)<<endl;
		
			error = 0.0;
			for(int i=0;i<f->n;i++) error += (prev[i]-x[i])*(prev[i]-x[i]);
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


void Nesterov::print_algo(){
	string algo = "Gradient Decent";
	cout<<algo;
}
