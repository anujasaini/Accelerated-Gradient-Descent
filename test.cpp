#include <bits/stdc++.h>
// #include "linear_eqns.h"
// #include "functor.h"
#include "gd.h"

using namespace std;

// to get the info of available code snippets : avail
// Sample Input : Paste here

int main(){

	Equation* e = new Equation(3, 3);

	CustomFunctor* c = new CustomFunctor(e);

	GD* gd  = new GD(0.01, 1000);

	float* ans = gd->minimize(c);

	cout<<ans[0]<<" "<<ans[1]<<" "<<ans[2];
}
