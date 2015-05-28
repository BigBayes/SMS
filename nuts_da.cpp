// The MEX version of nuts_da.m by Matthew D. Hoffman.
// Note that the output samples are stored in COLUMNS (rather than in ROWS as is with 
// the original Matlab code) due to performance considerations.
// 
// to compile: mex -output nuts_da_mex CXXFLAGS="$CXXFLAGS -std=c++0x" nuts_da.cpp
// or should the above line fail, try:
//             mex -output nuts_da_mex CXXFLAGS="$CXXFLAGS -std=c++11 -stdlib=libc++" nuts_da.cpp
// 
// Minjie Xu (chokkyvista06@gmail.com)

#include "mex.h"
#include "matrix.h"

#include <cmath>
#include <random>
#include <stdio.h>
#include <string.h>
#include <time.h>

double find_reasonable_epsilon(const double *theta0, const double *grad, const double logp, const mxArray *f,
							   const int D, std::mt19937 &generator, std::normal_distribution<double> &stdnrm,
							   double *r0, mxArray *thetaprimemat, double *rprime, double *gradprime);

void build_tree(const double *theta, const double *r, const double *grad, 
				const double logu, const bool v, const int j, const double epsilon, const mxArray *f, const double joint0, 
				const int D, std::mt19937 &generator, std::uniform_real_distribution<double> &uni01, 
				double *thetaminus, double *rminus, double *gradminus, double *thetaplus, double *rplus, double *gradplus, 
				mxArray *thetaprimemat, double *rprime, double *gradprime, double *logpprime,
				double *nprime, bool *sprime, double *alphaprime, double *nalphaprime);

bool stop_criterion(const double *thetaminus, const double *thetaplus, const double *rminus, const double *rplus, const int D);

//function [samples, epsilon] = nuts_da(f, M, Madapt, theta0, delta)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	if (nrhs < 4) {
		mexErrMsgTxt("Expecting at least the first 4 parameters.");
	}
    if (!mxIsClass(prhs[0], "function_handle")) {
		mexErrMsgTxt("The 1st input argument should be a function handle.");
	}

//	assert(size(theta0, 1) == 1);
	if (mxGetM(prhs[3]) != 1) {
		mexErrMsgTxt("The 4th input argument should be a row vector.");
	}
	
	mxArray *f = const_cast<mxArray *>(prhs[0]);
	int M = (int)mxGetScalar(prhs[1]);
	int Madapt = (int)mxGetScalar(prhs[2]);
	double *theta0 = mxGetPr(prhs[3]);

//	if (nargin < 5)
//		delta = 0.6;
//	end
	double delta = nrhs < 5 ? 0.6 : mxGetScalar(prhs[4]);

//	D = length(theta0);
	int D = (int)mxGetN(prhs[3]);
//	samples = zeros(D, M+Madapt);
	double *samples = new double [D*(M+Madapt)];

//	[logp, grad] = f(theta0);
	mxArray *lhs[2], *rhs[2] = {f, const_cast<mxArray *>(prhs[3])};
	mexCallMATLAB(2, lhs, 2, rhs, "feval");
	double logp = mxGetScalar(lhs[0]);
	double *grad = new double [D];
	const size_t sz = D*sizeof(double);
	memcpy(grad, mxGetPr(lhs[1]), sz);
	mxDestroyArray(lhs[0]);
	mxDestroyArray(lhs[1]);
  
//	samples(:, 1) = theta0';
	memcpy(samples, theta0, sz);

	std::mt19937 generator((unsigned int)time(0));
	std::normal_distribution<double> stdnrm(0.0, 1.0);
	std::exponential_distribution<double> exp1(1.0);
	std::uniform_real_distribution<double> uni01(0.0, 1.0);
  
//	epsilon = find_reasonable_epsilon(theta0, grad, logp, f);
	mxArray *thetaprimemat = mxCreateDoubleMatrix(1, D, mxREAL);
	double *thetaprime = mxGetPr(thetaprimemat);
	double *rprime = new double [D];
	double *gradprime = new double [D];
	double *r0 = new double [D];
	double epsilon = find_reasonable_epsilon(theta0, grad, logp, f, D, generator, stdnrm, 
											 r0, thetaprimemat, rprime, gradprime);

//	gamma = 0.05;
//	t0 = 10;
//	kappa = 0.75;
//	mu = log(10*epsilon);
	const double gamma = 0.05;
	const double t0 = 10.0;
	const double kappa = 0.75;
	const double mu = log(10.0 * epsilon);
//	epsilonbar = 1;
//	Hbar = 0;
	double epsilonbar = 1.0;
	double Hbar = 0.0;

	double *thetaminus = new double [D];
	double *thetaplus = new double [D];
	double *rminus = new double [D];
	double *rplus = new double [D];
	double *gradminus = new double [D];
	double *gradplus = new double [D];
	double logpprime, nprime, alpha, nalpha;
	bool sprime;
	
	double *samples_ptr = samples;
	
//	for m = 2:M+Madapt,
	for (int m = 2; m <= M+Madapt; ++m) {
//		r0 = randn(1, D);
//		joint = logp - 0.5 * (r0 * r0');
		double joint = 0.0;
		for (int i = 0; i < D; ++i) {
			double r0i = stdnrm(generator);
			r0[i] = r0i;
			joint += r0i * r0i;
		}
		joint = logp - 0.5 * joint;
//		logu = joint - exprnd(1);
		double logu = joint - exp1(generator);
//		thetaminus = samples(:, m-1)';
//		thetaplus = thetaminus;
//		rminus = r0;
//		rplus = r0;
//		gradminus = grad;
//		gradplus = grad;
		memcpy(thetaminus, samples_ptr, sz);
		memcpy(thetaplus, samples_ptr, sz);
		memcpy(rminus, r0, sz);
		memcpy(rplus, r0, sz);
		memcpy(gradminus, grad, sz);
		memcpy(gradplus, grad, sz);	
//		j = 0;
//		samples(:, m) = samples(:, m-1);
//		n = 1;
		int j = 0;
		memcpy(samples_ptr+D, samples_ptr, sz);
		double n = 1;
		
		samples_ptr += D;
		
//		s = 1;
//		while (s == 1)
//			v = 2*(rand() < 0.5)-1;
		bool s = true;
		while (s) {
			bool v = uni01(generator) < 0.5;
//			if (v == -1)
//				[thetaminus, rminus, gradminus, ~, ~, ~, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha] = ...
//					build_tree(thetaminus, rminus, gradminus, logu, v, j, epsilon, f, joint);
//			else
//				[~, ~, ~, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alpha, nalpha] = ...
//					build_tree(thetaplus, rplus, gradplus, logu, v, j, epsilon, f, joint);
//			end
			if (v == false) {
				build_tree(thetaminus, rminus, gradminus, logu, v, j, epsilon, f, joint, D, generator, uni01, 
						   thetaminus, rminus, gradminus, NULL, NULL, NULL, thetaprimemat, rprime, gradprime, 
						   &logpprime, &nprime, &sprime, &alpha, &nalpha);
			} else {
				build_tree(thetaplus, rplus, gradplus, logu, v, j, epsilon, f, joint, D, generator, uni01, 
						   NULL, NULL, NULL, thetaplus, rplus, gradplus, thetaprimemat, rprime, gradprime, 
						   &logpprime, &nprime, &sprime, &alpha, &nalpha);
			}
			
//			if ((sprime == 1) && (rand() < nprime/n))
//				samples(:, m) = thetaprime';
//				logp = logpprime;
//				grad = gradprime;
//			end
			if (sprime && uni01(generator) < nprime/n) {
				memcpy(samples_ptr, thetaprime, sz);
				logp = logpprime;
				memcpy(grad, gradprime, sz);
			}
//			n = n + nprime;
//			s = sprime && stop_criterion(thetaminus, thetaplus, rminus, rplus);
//			j = j + 1;
//		end
			n += nprime;
			s = sprime && stop_criterion(thetaminus, thetaplus, rminus, rplus, D);
			j += 1;
		}
//		eta = 1 / (m - 1 + t0);
//		Hbar = (1 - eta) * Hbar + eta * (delta - alpha / nalpha);
		double eta = 1.0 / (m - 1 + t0);
		Hbar = (1.0 - eta) * Hbar + eta * (delta - alpha / nalpha);
//		if (m <= Madapt)
//			epsilon = exp(mu - sqrt(m-1)/gamma * Hbar);
//			eta = (m-1)^-kappa;
//			epsilonbar = exp((1 - eta) * log(epsilonbar) + eta * log(epsilon));
//		else
//			epsilon = epsilonbar;
//		end
		if (m <= Madapt) {
			epsilon = exp(mu - sqrt(m-1.0)/gamma * Hbar);
			eta = pow(m-1.0, -kappa);
			epsilonbar = exp((1.0 - eta) * log(epsilonbar) + eta * log(epsilon));
		} else {
			epsilon = epsilonbar;
		}
//	end
	}
//	samples = samples(:, Madapt+1:end)';
	plhs[0] = mxCreateDoubleMatrix(D, M, mxREAL);
	memcpy(mxGetPr(plhs[0]), samples+Madapt*D, M*sz);
	
	
	delete [] gradplus;
	delete [] gradminus;
	delete [] rplus;
	delete [] rminus;
	delete [] thetaplus;
	delete [] thetaminus;
	
	delete [] r0;
	delete [] gradprime;
	delete [] rprime;
	mxDestroyArray(thetaprimemat);
	
	delete [] grad;
	delete [] samples;
}

//function [thetaprime, rprime, gradprime, logpprime] = leapfrog(theta, r, grad, epsilon, f)
void leapfrog(const double *theta, const double *r, const double *grad, const double epsilon, const mxArray *f, const int D, 
			  mxArray *thetaprimemat, double *rprime, double *gradprime, double *logpprime) {
//	rprime = r + 0.5 * epsilon * grad;
//	thetaprime = theta + epsilon * rprime;
	double *thetaprime = mxGetPr(thetaprimemat);
	for (int i = 0; i < D; ++i) {
		rprime[i] = r[i] + 0.5 * epsilon * grad[i];
		thetaprime[i] = theta[i] + epsilon * rprime[i];
	}
//	[logpprime, gradprime] = f(thetaprime);
	mxArray *lhs[2], *rhs[2] = {const_cast<mxArray *>(f), thetaprimemat};
	mexCallMATLAB(2, lhs, 2, rhs, "feval");
	*logpprime = mxGetScalar(lhs[0]);
	memcpy(gradprime, mxGetPr(lhs[1]), D*sizeof(double));
	mxDestroyArray(lhs[0]);
	mxDestroyArray(lhs[1]);
//	rprime = rprime + 0.5 * epsilon * gradprime;
	for (int i = 0; i < D; ++i) {
		rprime[i] += 0.5 * epsilon * gradprime[i];
	}
//end
}

//function criterion = stop_criterion(thetaminus, thetaplus, rminus, rplus)
bool stop_criterion(const double *thetaminus, const double *thetaplus, const double *rminus, const double *rplus, const int D) {
//	thetavec = thetaplus - thetaminus;
//	criterion = (thetavec * rminus' >= 0) && (thetavec * rplus' >= 0);
	double term1 = 0.0, term2 = 0.0;
	for (int i = 0; i < D; ++i) {
		double thetadiff = thetaplus[i] - thetaminus[i];
		term1 += thetadiff * rminus[i];
		term2 += thetadiff * rplus[i];
	}
	return term1 >= 0 && term2 >= 0;
//end
}

//function [thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime] = ...
//				build_tree(theta, r, grad, logu, v, j, epsilon, f, joint0)
void build_tree(const double *theta, const double *r, const double *grad, 
				const double logu, const bool v, const int j, const double epsilon, const mxArray *f, const double joint0, 
			    const int D, std::mt19937 &generator, std::uniform_real_distribution<double> &uni01, 
			    double *thetaminus, double *rminus, double *gradminus, double *thetaplus, double *rplus, double *gradplus, 
			    mxArray *thetaprimemat, double *rprime, double *gradprime, double *logpprime, 
			    double *nprime, bool *sprime, double *alphaprime, double *nalphaprime) {
	double *thetaprime = mxGetPr(thetaprimemat);
	const size_t sz = D*sizeof(double);
	
//	if (j == 0)
	if (j == 0) {
//		[thetaprime, rprime, gradprime, logpprime] = leapfrog(theta, r, grad, v*epsilon, f);
		leapfrog(theta, r, grad, (v ? epsilon : -epsilon), f, D, thetaprimemat, rprime, gradprime, logpprime);
//		joint = logpprime - 0.5 * (rprime * rprime');
		double joint = 0.0;
		for (int i = 0; i < D; ++i) {
			joint += rprime[i]*rprime[i];
		}
		joint = *logpprime - 0.5 * joint;
//		nprime = logu < joint;
		*nprime = logu < joint;
//		sprime = logu - 1000 < joint;
		*sprime = (logu - 1000.0) < joint;
//		thetaminus = thetaprime;
//		thetaplus = thetaprime;
//		rminus = rprime;
//		rplus = rprime;
//		gradminus = gradprime;
//		gradplus = gradprime;
		if (thetaminus) memcpy(thetaminus, thetaprime, sz);
		if (rminus) memcpy(rminus, rprime, sz);
		if (gradminus) memcpy(gradminus, gradprime, sz);
		if (thetaplus) memcpy(thetaplus, thetaprime, sz);
		if (rplus) memcpy(rplus, rprime, sz);
		if (gradplus) memcpy(gradplus, gradprime, sz);
//		alphaprime = min(1, exp(logpprime - 0.5 * (rprime * rprime') - joint0));
		*alphaprime = std::min(1.0, exp(joint - joint0));
//		nalphaprime = 1;
		*nalphaprime = 1.0;
//	else
	} else {
//		[thetaminus, rminus, gradminus, thetaplus, rplus, gradplus, thetaprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime] = ...
//					build_tree(theta, r, grad, logu, v, j-1, epsilon, f, joint0);
		double *thetaminusout = new double [D];
		double *rminusout = new double [D];
		double *gradminusout = new double [D];
		double *thetaplusout = new double [D];
		double *rplusout = new double [D];
		double *gradplusout = new double [D];
		build_tree(theta, r, grad, logu, v, j-1, epsilon, f, joint0, D, generator, uni01, 
				   thetaminusout, rminusout, gradminusout, thetaplusout, rplusout, gradplusout, 
				   thetaprimemat, rprime, gradprime, logpprime, nprime, sprime, alphaprime, nalphaprime);
//		if (sprime == 1)
		if (*sprime == true) {
			double *thetaprimecpy = new double [D];
			memcpy(thetaprimecpy, thetaprime, sz);
			double *gradprimecpy = new double [D];
			memcpy(gradprimecpy, gradprime, sz);
			double logpprimecpy = *logpprime;
			double nprimecpy = *nprime;
			bool sprimecpy = *sprime;
			double alphaprimecpy = *alphaprime;
			double nalphaprimecpy = *nalphaprime;
//			if (v == -1)
			if (v == false) {
//				[thetaminus, rminus, gradminus, ~, ~, ~, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2] = ...
//					build_tree(thetaminus, rminus, gradminus, logu, v, j-1, epsilon, f, joint0);
				build_tree(thetaminusout, rminusout, gradminusout, logu, v, j-1, epsilon, f, joint0, D, generator, uni01, 
						   thetaminusout, rminusout, gradminusout, NULL, NULL, NULL, thetaprimemat, rprime, gradprime, 
						   logpprime, nprime, sprime, alphaprime, nalphaprime);
//			else
			} else {
//				[~, ~, ~, thetaplus, rplus, gradplus, thetaprime2, gradprime2, logpprime2, nprime2, sprime2, alphaprime2, nalphaprime2] = ...
//					build_tree(thetaplus, rplus, gradplus, logu, v, j-1, epsilon, f, joint0);
				build_tree(thetaplusout, rplusout, gradplusout, logu, v, j-1, epsilon, f, joint0, D, generator, uni01, 
						   NULL, NULL, NULL, thetaplusout, rplusout, gradplusout, thetaprimemat, rprime, gradprime, 
						   logpprime, nprime, sprime, alphaprime, nalphaprime);
//			end
			}
//			if (rand() < nprime2 / (nprime + nprime2))
			if (uni01(generator) < *nprime / (nprimecpy + *nprime)) {
//				thetaprime = thetaprime2;
//				gradprime = gradprime2;
//				logpprime = logpprime2;
			}
//			end
			else {
				memcpy(thetaprime, thetaprimecpy, sz);
				memcpy(gradprime, gradprimecpy, sz);
				*logpprime = logpprimecpy;
			}
//			nprime = nprime + nprime2;
//			sprime = sprime && sprime2 && stop_criterion(thetaminus, thetaplus, rminus, rplus);
//			alphaprime = alphaprime + alphaprime2;
//			nalphaprime = nalphaprime + nalphaprime2;
			*nprime += nprimecpy;
			*sprime = sprimecpy && *sprime && stop_criterion(thetaminusout, thetaplusout, rminusout, rplusout, D);
			*alphaprime += alphaprimecpy;
			*nalphaprime += nalphaprimecpy;
			
			delete [] gradprimecpy;
			delete [] thetaprimecpy;
//		end
		}
		
		if (thetaminus) memcpy(thetaminus, thetaminusout, sz);
		if (rminus) memcpy(rminus, rminusout, sz);
		if (gradminus) memcpy(gradminus, gradminusout, sz);
		if (thetaplus) memcpy(thetaplus, thetaplusout, sz);
		if (rplus) memcpy(rplus, rplusout, sz);
		if (gradplus) memcpy(gradplus, gradplusout, sz);
		delete [] thetaminusout;
		delete [] rminusout;
		delete [] gradminusout;
		delete [] thetaplusout;
		delete [] rplusout;
		delete [] gradplusout;
//	end
	}
//end
}

//function epsilon = find_reasonable_epsilon(theta0, grad0, logp0, f)
double find_reasonable_epsilon(const double *theta0, const double *grad0, const double logp0, const mxArray *f, 
							   const int D, std::mt19937 &generator, std::normal_distribution<double> &stdnrm, 
							   double *r0, mxArray *thetaprimemat, double *rprime, double *gradprime) {
//	epsilon = 1;
	double epsilon = 1.0;
//	r0 = randn(1, length(theta0));
	for (int i = 0; i < D; ++i) {
		r0[i] = stdnrm(generator);
	}
//	[~, rprime, ~, logpprime] = leapfrog(theta0, r0, grad0, epsilon, f);
	double logpprime;
	leapfrog(theta0, r0, grad0, epsilon, f, D, thetaprimemat, rprime, gradprime, &logpprime);
//	acceptprob = exp(logpprime - logp0 - 0.5 * (rprime * rprime' - r0 * r0'));
	double acceptprob = 0.0;
	for (int i = 0; i < D; ++i) {
		acceptprob += (r0[i] + rprime[i]) * (r0[i] - rprime[i]);
	}
	acceptprob = exp(logpprime - logp0 + 0.5 * acceptprob);
//	a = 2 * (acceptprob > 0.5) - 1;
	bool a = acceptprob > 0.5;
	const double twopowa = (a ? 2 : 0.5);
//	while (acceptprob^a > 2^(-a))
	double acceptprobpowa = (a ? acceptprob : 1.0/acceptprob);
	while (acceptprobpowa * twopowa > 1) {
//		epsilon = epsilon * 2^a;
		epsilon *= twopowa;
//		[~, rprime, ~, logpprime] = leapfrog(theta0, r0, grad0, epsilon, f);
		leapfrog(theta0, r0, grad0, epsilon, f, D, thetaprimemat, rprime, gradprime, &logpprime);
//		acceptprob = exp(logpprime - logp0 - 0.5 * (rprime * rprime' - r0 * r0'));
		acceptprob = 0.0;
		for (int i = 0; i < D; ++i) {
			acceptprob += (r0[i] + rprime[i]) * (r0[i] - rprime[i]);
		}
		acceptprob = logpprime - logp0 + 0.5 * acceptprob;
		acceptprobpowa = (a ? exp(acceptprob) : exp(-acceptprob));
//	end
	}
//end
  	return epsilon;
}
