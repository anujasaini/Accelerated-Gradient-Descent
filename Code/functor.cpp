#include "functor.h"
#include "utils.h"

CustomFunctor::CustomFunctor(float** A, float* b, int n, int m){
	this->n = n;
	this->m = m;
	this->A = A;
	this->b = b;
}

CustomFunctor::CustomFunctor(Equation* e){
	this->m = e->m;
	this->n = e->n;
	this->A = e->A;
	this->b = e->b;
}

float CustomFunctor::eval(float* x){
	int s1 = this->m;
	int s2 = this->n;
	return dot_product(sub(matrix_vector_mult(this->A,x,s1,s2,s2),this->b,s1,s1), sub(matrix_vector_mult(this->A,x,s1,s2,s2),this->b,s1,s1), s1, s1);	
}

float* CustomFunctor::grad(float* x){
	int s1 = this->m;
	int s2 = this->n;
	float* ans = new float[3]{0};
	float** A_T = matrix_transpose(this->A, s1, s2);
	float* Ax = matrix_vector_mult(this->A, x, s1, s2, s2);
	float* Ax_minus_b = sub(Ax, this->b ,s1,s1);
	float* A_T_Ax_minus_b = matrix_vector_mult(A_T, Ax_minus_b, s2, s1, s1);
	ans = scalar_mult(A_T_Ax_minus_b, 2.0, s2);
	
	return ans;
}
