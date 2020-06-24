
#ifndef _utils_h
#define _utils_h

#include <bits/stdc++.h>

using namespace std;


float dot_product(float* v1, float* v2, int n , int m );

float** matrix_mult(float** m1, float** m2, int s1, int s2, int s3, int s4);

float* matrix_vector_mult(float** A, float* x, int s1, int s2, int s3);

float** matrix_transpose(float** m, int s1, int s2);

float* add(float* v1, float* v2, int n, int m);

float* scalar_mult(float* v1, float k, int n);

float* sub(float* v1, float* v2, int n, int m);

float** allocate_memory(int n, int m);

float determinant(float** matrix, int n, int m);

void getCofactor(float** A, float** temp, int n, int p, int q) ;

void adjoint(float** A,float** adj, int N); 

float** inverse(float** A, int N); 


#endif