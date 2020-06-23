#include "utils.h"

using namespace std;

float dot_product(float* v1, float* v2, int n , int m ){

	assert(n == m);
	float result = 0;
	for(int i=0;i<n;i++){
		result += v1[i]*v2[i];
	}
	return result;
}


float** matrix_mult(float** m1, float** m2, int s1, int s2, int s3, int s4){


	assert(s2 == s3);
	
	float** m = new float*[s1];

	for(int i=0;i<s1;i++) m[i] = new float[s4];

	for(int i=0;i<s1;i++){
		for(int j=0;j<s4;j++){
			float val = 0;
			for(int k=0;k<s2;k++){
				val += m1[i][k]*m2[k][j];
			}
			m[i][j] = val;
		}
	}

}

float* matrix_vector_mult(float** A, float* x, int s1, int s2, int s3){

	assert(s2 == s3);

	float* result = new float[s1];

	for(int i=0;i<s1;i++){
		
		float temp = 0;
		for(int j=0;j<s2;j++){
			temp += A[i][j]*x[j];
		}

		result[i] = temp;
	}

	return result;

}


float** matrix_transpose(float** m, int s1, int s2){

	float** ret = new float*[s2];

	for(int i=0;i<s2;i++) ret[i] = new float[s1];
	
	for(int i=0;i<s2;i++){
		for(int j=0;j<s1;j++){
			ret[i][j] = m[j][i];
		}
	}
	return ret;
}


float* add(float* v1, float* v2, int n, int m){

	assert(n ==m);

	float* result = new float[n];
	for(int i=0;i<n;i++)
		result[i] = (v1[i] + v2[i]);
	return result;
}


float* scalar_mult(float* v1, float k, int n){

	float* result = new float[n];
	for(int i=0;i<n;i++)
		result[i] = (v1[i]*k);
	return result;
}


float* sub(float* v1, float* v2, int n, int m){
	assert(n ==m);

	float* result = new float[n];
	for(int i=0;i<n;i++)
		result[i] = (v1[i] - v2[i]);
	return result;
}


float** allocate_memory(int n, int m){
	float** ret = new float*[n];
	for(int i=0;i<n;i++)
		ret[i] = new float[m];

	return ret;
}

float determinant(float** matrix, int n, int m) {
	assert(n==m);   
	
	float det = 0;
	


	float** submatrix = new float*[n-1];
	for(int i=0;i<n-1;i++){
		submatrix[i] = new float[n-1];
	}

	if (n == 2)
		return ((matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1]));
	else {
		for (int x = 0; x < n; x++) {
			int subi = 0; 
			for (int i = 1; i < n; i++) {
				int subj = 0;
				for (int j = 0; j < n; j++) {
					if (j == x)
						continue;
					submatrix[subi][subj] = matrix[i][j];
					subj++;
				}
				subi++;
			}
			det = det + (pow(-1, x) * matrix[0][x] * determinant( submatrix , n - 1 , n - 1 ));
		}
	}
	
	return det;
}


void getCofactor(float** A, float** temp, int n, int p, int q) 
{ 
    int i = 0, j = 0; 
  
    // Looping for each element of the matrix 
    for (int row = 0; row < n; row++) 
    { 
        for (int col = 0; col < n; col++) 
        { 
            //  Copying into temporary matrix only those element 
            //  which are not in given row and column 
            if (row != p && col != q) 
            { 
                temp[i][j++] = A[row][col]; 
  
                // Row is filled, so increase row index and 
                // reset col index 
                if (j == n - 1) 
                { 
                    j = 0; 
                    i++; 
                } 
            } 
        } 
    } 
} 



void adjoint(float** A,float** adj, int N) 
{ 
    if (N == 1) 
    { 
        adj[0][0] = 1; 
        return; 
    } 
  
    // temp is used to store cofactors of A[][] 
    int sign = 1;
    float** temp = allocate_memory(N,N); 
  
    for (int i=0; i<N; i++) 
    { 
        for (int j=0; j<N; j++) 
        { 
            // Get cofactor of A[i][j]

            getCofactor(A, temp, N, i, j); 
  
            // sign of adj[j][i] positive if sum of row 
            // and column indexes is even. 
            sign = ((i+j)%2==0)? 1: -1; 
  
            // Interchanging rows and columns to get the 
            // transpose of the cofactor matrix 
            adj[j][i] = (sign)*(determinant(temp, N-1, N-1)); 
        } 
    } 
} 
  
// Function to calculate and store inverse, returns false if 
// matrix is singular 
float** inverse(float** A, int N) 
{ 
    // Find determinant of A[][] 
    float** inverse = allocate_memory(N,N);

    float det = determinant(A, N, N); 
    if (det == 0) 
    { 
        cout << "Singular matrix, can't find its inverse"; 
        return NULL; 
    } 
  
    // Find adjoint 
    float** adj = allocate_memory(N,N); 
    adjoint(A, adj, N); 
  
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
    for (int i=0; i<N; i++) 
        for (int j=0; j<N; j++) 
            inverse[i][j] = adj[i][j]/float(det); 
  
    return inverse; 
} 