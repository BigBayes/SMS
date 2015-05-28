%% Model configurations
n = 1e3;
d = 5;
m = 8;
di = mat2cell(1:n, 1, [ones(1,m-1)*floor(n/m), n-(m-1)*floor(n/m)]);

%% Generate MODEL and DATA
P = rand(d);
model = bayeslogreg(d, rand(d,1), P*P');
[data, model] = bayeslogreg_gendata(n, d, model);
approxfam = approxfam_gaussian();

%% Sampler configurations
sampler = sampler_nuts_da(20*d, 2);
T = 1e3;
dampalpha = 0.2;
partprior = false;
async = false;
[~, tsmpls] = smssample(model, data, {1:n}, approxfam, false, false, 0, sampler, rand(100,d), 1000*d, 1);
tsmpls = tsmpls{1};

%% SMS sampling
[qparam, smpls, divs, errs, lsm] = smssample(model, data, di, approxfam, async, partprior, dampalpha, sampler, tsmpls, T);

%% Visualize the results
vislsm(lsm, mean(tsmpls));
viserrs(m, 1, 1, {{errs}}, {'SMS'});