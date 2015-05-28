function [samples] = spikeslab_gibbs(model, burnin, T, StoreEvery, sampler_param)
% [samples] = spikeslab_gibbs(model, burnin, T, StoreEvery, sampler_param)
% spikeslab_gibbs returns samples of w and s from a paired Gibbs sampler 
% see paper by Titsias et. al for further details
%
% Michalis Titsias's code has been modified to use a different prior for w:
% the prior is p(w, s) = \prod_d p(w_d|s_d) p(s_d)
% p(s_d) is parametrized by logodds called logAlpha (natural parameter)
% p(w_d|s_d=1) is a gaussian with non-zero mean mu_1 and variance sigma_1^2
% - the natural parameters are invsigma=sigma_1^{-2} and
% - invsigmamu=invsigma * mu_1
% p(w_d|s_d=0) is a gaussian with zero mean and variance wvariance
%
% Note:
% - w is not sampled when s_d=0 since the distribution is known
% - the code assumes fixed hyperparameters
% - varianceratio = noisevariance * invsigma;


num_stored = floor(T/StoreEvery);
samples.W = zeros(num_stored, model.dim);
samples.S = zeros(num_stored, model.dim);

% initialize S and W
S = rand(1, model.dim) <= sigmoid(sampler_param.logAlpha'); % sample from prior
W = (pinv(sampler_param.XtX) * sampler_param.XtY)'; % better initialization?

diagXtX = diag(sampler_param.XtX);
xxsigma = diagXtX + sampler_param.varianceRatio;
logxxsigma = log(xxsigma);

sqrt_sigma2_xxsigma = sqrt(model.param.noisevariance ./ xxsigma);
half_mu_invsigmamu = 0.5*(sampler_param.invsigmamu.^2) ./ sampler_param.invsigma;
noisevariance_invsigmamu = sampler_param.invsigmamu * model.param.noisevariance;

% start the Gibbs sampling
cnt = 0;
for it=1:(burnin + T)

    for d=randperm(model.dim)
        
        setMinus = S;
        setMinus(d) = logical(0);
        
        b = sum(W(setMinus) .* sampler_param.XtX(d,setMinus), 2);
        
        diff_term = sampler_param.XtY(d) - b + noisevariance_invsigmamu(d);
        
        um = sampler_param.logAlpha(d) + sampler_param.halflogVarianceRatio(d) ...
            - 0.5*logxxsigma(d) - half_mu_invsigmamu(d) ...
            +  (0.5/model.param.noisevariance)*((diff_term^2)/xxsigma(d));
        
        % sample binary variable
        S(d) = logical( rand < (1/(1+exp(-um))) );
        
        % if s_m=1 sample also the parameter w_m
        if S(d) == 1
            W(d) = diff_term/xxsigma(d) + sqrt_sigma2_xxsigma(d)*randn;
        end
        
    end
    
    % after burnin keep the samples
    if (it > burnin) && (mod(it,StoreEvery) == 0)
        cnt = cnt + 1;
        samples.W(cnt,:) = W;
        samples.S(cnt,:) = S;
    end
end