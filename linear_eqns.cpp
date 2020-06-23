#include <bits/stdc++.h>
#include "linear_eqns.h"
#include "utils.h"

using namespace std;


// m equations in n variables with random coeficients 
Equation::Equation(int m, int n){
	float** coeffs = new float*[m];
	for(int i=0;i<m;i++) coeffs[i] = new float[n];

	float* b = new float[m];

	do{
		
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				float t1 = rand()%100;
				coeffs[i][j] = t1;
			}
			float t2 = rand()%100;
			b[i] = t2;
		}

	}while(determinant(coeffs,n,m)==0);

	float max_A = INT_MIN;

	float det_A = determinant(coeffs, n, n);
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			if(coeffs[i][j] > max_A) max_A = coeffs[i][j];
		}
	}

	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			coeffs[i][j] = coeffs[i][j]/max_A;
		}

		b[i] = b[i]/max_A;
	}



	this->A = coeffs;
	this->b = b;
	this->m = m;
	this->n = n;

	this->stringify();
}

// m equations in n variables with coeficients as given by 2D array
Equation::Equation(float** coeffs, float* b,  int n, int m){

	this->A = coeffs;
	this->b = b;
	this->m = m;
	this->n = n;

	this->stringify();
}

float* Equation::evaluate(float* values){

	float* ret = new float[m];
	for(int i=0;i<this->m;i++){
		float value = 0.0;
		for(int j=0;j<this->n;j++){
			value += this->A[i][j]*values[j];
		}

		value -= this->b[i];
		ret[i] = value;
	}

	return ret;
}

	


void Equation::stringify(){

	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			// if(this->A[i][j]==0) continue;
			cout<<this->A[i][j]<<"x"<<j<<" ";
			if(j!=n-1) cout<<" + ";
		}
		cout<<" = "<<this->b[i]<<endl;
	}
}



