#include "linear_eqns.h"
#include "functor.h"
#include "utils.h"

#include "gd.h"
#include "nesterov.h"
#include "heavy_ball.h"
#include "fista.h"

using namespace std;

// to get the info of available code snippets : avail
// Sample Input : Paste here

int main(){

	

	int n = 3;
	float lr = 0.01;
	int steps = 100000;
	float momentum_coeff = 1;
	for(int n =3;n<4;n++)
	{
		Equation* e = new Equation(n, n);
		CustomFunctor* c = new CustomFunctor(e);
		float* orig_sol = matrix_vector_mult(inverse(e->A, n), e->b, n, n , n);
		

		// gradient decent
		clock_t begin = clock();
		cout<<"Gradient decent method test \n\n";

		GD* gd  = new GD(lr, steps);
		float* ans = gd->solve(c);
		
		cout<<endl<<"Variable\tOriginal\tObtained\n";
		for(int i=0;i<n;i++)
			cout<<"x"<<i<<"\t\t"<<orig_sol[i]<<"\t\t"<<ans[i]<<endl;

		clock_t end = clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

		cout<<"Time taken :"<<elapsed_secs<<endl<<"\n\n";


		// heavy ball test 

		begin =clock();				
		cout<<"HeavyBall method test \n\n";
		HeavyBall* hb  = new HeavyBall(lr, 0.1, steps );
		ans = hb->solve(c);

		cout<<endl<<"Variable\tOriginal\tObtained\n";
		for(int i=0;i<n;i++)
			cout<<"x"<<i<<"\t\t"<<orig_sol[i]<<"\t\t"<<ans[i]<<endl;

		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

		cout<<"Time taken :"<<elapsed_secs<<endl<<"\n\n";


		// nestrov test 

		begin =clock();				
		cout<<"Nesterov method test \n\n";
		Nesterov* nestrov  = new Nesterov(lr, momentum_coeff, steps );
		ans = nestrov->solve(c);

		cout<<endl<<"Variable\tOriginal\tObtained\n";
		for(int i=0;i<n;i++)
			cout<<"x"<<i<<"\t\t"<<orig_sol[i]<<"\t\t"<<ans[i]<<endl;

		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

		cout<<"Time taken :"<<elapsed_secs<<endl<<"\n\n";


		//  FISTA test 

		begin =clock();				
		cout<<"FISTA method test \n\n";
		Fista* fista  = new Fista(lr, momentum_coeff, steps );
		ans = fista->solve(c);
		cout<<endl<<"Variable\tOriginal\tObtained\n";
		for(int i=0;i<n;i++)
			cout<<"x"<<i<<"\t\t"<<orig_sol[i]<<"\t\t"<<ans[i]<<endl;

		end = clock();
		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

		cout<<"Time taken :"<<elapsed_secs<<endl<<"\n\n";

	}
}
				