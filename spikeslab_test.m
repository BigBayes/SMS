% Bayesian sparse Linear Regression model with spike and slab prior

%% Configurations
clear all; close all; clc;

nrun = 1;
mm = [2]; % mm is a vector of different values of m (# partitions)


%% Generate MODEL and DATA
%dataset = 'toy';
dataset = 'boston-housing';
switch dataset
    case 'toy'
        n = 4e2;
        d = 5;
        pi = 0.5;
        noisevariance = 0.01;
        wvariance = 1;
        mu = 5*randn(d,1);
        model = spikeslab(d, mu, wvariance, pi, noisevariance);
        fprintf('printing s and w\n');
        model.param.s'
        model.param.w'
        [x, y] = spikeslab_gendata(n, model);
    case 'boston-housing'
        data_tmp = load('dataset/data_boston.mat');
        x = data_tmp.X'; y = data_tmp.y';
        n = length(y);
        d = size(x, 1);
        pi = 0.5;
        noisevariance = var(y) * 0.1;
        wvariance = 2;
        mu = zeros(d,1);
        model = spikeslab(d, mu, wvariance, pi, noisevariance);
end
data = struct('x', num2cell(x,1), 'y', num2cell(y));

switch dataset
    case 'boston-housing'
        gndtruth = load('dataset/boston_groundtruth_gibbs.mat');
        samples_gndtruth =  gndtruth.samples;
        param_gndtruth = struct('s', mean(gndtruth.samples.S, 1), ...
            'w', mean(gndtruth.samples.S .* gndtruth.samples.W, 1));
        w_gndtruth = gndtruth.truesw;
        fprintf('printing s and w\n');
        param_gndtruth.s
        param_gndtruth.w
        % s = [1.0000    1.0000    0.0260    0.9998    1.0000    1.0000  ...
        %     0.0172    1.0000    1.0000    0.9998    1.0000    0.9998  1.0000]
        % w = [-0.1056   0.1164    0.0005    0.0795   -0.2546    0.3179 ...
        %      -0.0000  -0.3428   0.2760   -0.1985   -0.2328    0.0860 -0.3921]
    case 'toy'
        samples_gndtruth = [];
        param_gndtruth = struct('w', model.param.w', 's', model.param.s');
    otherwise
       error('ground truth not available; use a long run of spikeslab_gibbs to generate ground truth');        
end


%% SMS
T = 1e3;
maxiter = 30;
burnin = ceil(T / 10);
thinning = 2;

async = false;
partprior = false;
dampalpha = 0.5;

approxfam = approxfam_spikeslab();
sampler = sampler_spikeslab(burnin, thinning, model);
    
for m = mm
    fprintf('m = %d\n', m);
    di = mat2cell(1:n, 1, [ones(1,m-1)*floor(n/m), n-(m-1)*floor(n/m)]);
    
    [qparam, smpls, divs, errs] = smssample(model, data, di, ...
        approxfam, async, partprior, dampalpha, sampler, ...
        samples_gndtruth, T, maxiter);
    
    fprintf('comparing true to estimated parameters\n');
    s_true = param_gndtruth.s
    s_est = 1 ./ (1 + exp(-qparam.logodds'))
    w_true = param_gndtruth.w
    w_est = qparam.invsigmamu' ./ qparam.invsigma' ./ (1 + exp(-qparam.logodds'))
    
end

