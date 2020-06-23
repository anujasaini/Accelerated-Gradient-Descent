#include <bits/stdc++.h>
#include "functor.h"
using namespace std;

class Optimizer{

public: 
	virtual vector<float> minimize(Functor f) = 0;

	virtual vector<float> maximize(Functor f) = 0;

	virtual void print_algo() = 0;
};

