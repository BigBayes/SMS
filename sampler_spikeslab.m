% Paired Gibbs sampler for Bayesian sparse linear regression with
% spike-and-slab prior on the weights
%
% see spikeslab_gibbs.m and the paper by Titsias et. al for further details


function sampler = sampler_spikeslab(burnin, thinning, model)
sampler.draw = @paired_gibbs;
sampler.burnin = burnin;
sampler.thinning = thinning;
% burnin and thinning can be handled in the underlying sampler 
% or within draw

    function smpls = paired_gibbs(approxfam, qxparam, factor, nsamples, smpl0)
        sampler_param.logAlpha = qxparam.logodds;
        sampler_param.varianceRatio = model.param.noisevariance * qxparam.invsigma;
        sampler_param.halflogVarianceRatio = 0.5*(log(qxparam.invsigma) ...
            + log(model.param.noisevariance));
        sampler_param.XtX = factor.data.XtX;
        sampler_param.XtY = factor.data.XtY;
        sampler_param.invsigma = qxparam.invsigma;
        sampler_param.invsigmamu = qxparam.invsigmamu;
        samples = spikeslab_gibbs(model, sampler.burnin, ...
            nsamples*sampler.thinning, sampler.thinning, sampler_param);
        smpls = struct('W', num2cell(samples.W, 2), 'S', num2cell(samples.S, 2));
        % thinning and burnin are handled within the sampler
    end

end