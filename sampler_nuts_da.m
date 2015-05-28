% The adaptive No-U-Turn sampler (nuts_da).
% For details, please refer to comments in 'smssample'.
% 
% Minjie Xu (chokkyvista06@gmail.com)

function sampler = sampler_nuts_da(burnin, thinning)
sampler.draw = @nuts_da_sample;
sampler.burnin = burnin;
sampler.thinning = thinning;

    function smpls = nuts_da_sample(approxfam, qxparam, factor, nsamples, smpl0)
%         smpls = nuts_da(@(w)hmcf(w,approxfam,qxparam,factor), ...
%             nsamples*sampler.thinning, sampler.burnin, smpl0);
%         smpls = smpls(1:sampler.thinning:end, :);
        smpls = nuts_da_mex(@(w)hmcf(w,approxfam,qxparam,factor), ...
            nsamples*sampler.thinning, sampler.burnin, smpl0);
        smpls = smpls(:, 1:sampler.thinning:end)';
    end

end


function [logp, grad] = hmcf(rw, approxfam, qxparam, factor)
cw = rw';
logp = approxfam.qklogpdf(rw, qxparam) + factor.logf(cw);
grad = (approxfam.qkgradlogpdf(rw, qxparam) + factor.gradlogf(cw))';
end