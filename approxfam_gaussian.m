% The multivariate-Gaussian approximating family.
% For details, please refer to comments in 'smssample'.
% 
% Minjie Xu (chokkyvista06@gmail.com)

function gaussaf = approxfam_gaussian()
% required fields
gaussaf.id = 'gaussian';
gaussaf.validparam = @gaussian_validparam;
gaussaf.emptyparam = @gaussian_emptyparam;
gaussaf.priorparam = @gaussian_priorparam;
gaussaf.mulall = @gaussian_mulall;
gaussaf.mul = @gaussian_mul;
gaussaf.div = @gaussian_div;
gaussaf.damp = @gaussian_damp;
gaussaf.diverge = @gaussian_diverge;
gaussaf.smpls2param = @gaussian_smpls2param;
gaussaf.param2smplm = @gaussian_param2smplm;
gaussaf.errmsr = @gaussian_errmsr;

% optional fields
gaussaf.qklogpdf = @gaussian_qklogpdf;
gaussaf.qkgradlogpdf = @gaussian_qkgradlogpdf;

end


function isvalid = gaussian_validparam(param)
if ~all(isfield(param, {'eta', 'omega'}))
    isvalid = false;
    return;
end
[~, p] = chol(param.omega);
isvalid = nnz(param.omega)==0 || (p <= 0);

end

function [param] = gaussian_emptyparam(model)
param.omega = zeros(model.dim);
param.eta = zeros(model.dim, 1);

end

function [param, exact] = gaussian_priorparam(model, m)
if nargin == 1
    m = 1;
end
switch model.id
    case {'bayeslogreg', 'gaussian'}
        param.omega = inv(model.param.sigma)./m;
        param.eta = (model.param.sigma\model.param.mu)./m;
        exact = true;
    case {'sparsebayeslogreg'}
        param.omega = diag(0.5./(model.param.b.^2.*m));
        param.eta = param.omega*model.param.mu;
        exact = false;
    otherwise
        error('approxfam ''gaussian'' does not yet support model ''%s''!', ...
            model.id);
end

end

function [param] = gaussian_mulall(params)
param = params(1);
flds = fieldnames(param);
for i = 2:numel(params)
    for j = 1:numel(flds)
        fld = flds{j};
        param.(fld) = param.(fld) + params(i).(fld);
    end
end

end

function [param] = gaussian_mul(param1, param2)
param = param1;
flds = fieldnames(param);
for i = 1:numel(flds)
    fld = flds{i};
    param.(fld) = param1.(fld) + param2.(fld);
end

end

function [param] = gaussian_div(param1, param2)
param = param1;
flds = fieldnames(param);
for i = 1:numel(flds)
    fld = flds{i};
    param.(fld) = param1.(fld) - param2.(fld);
end

end

function [param] = gaussian_damp(nparam, oparam, dampalpha)
param = nparam;
flds = fieldnames(param);
for i = 1:numel(flds)
    fld = flds{i};
    if isstruct(dampalpha)
        talpha = dampalpha.(fld);
    else
        talpha = dampalpha;
    end
    param.(fld) = (1-talpha)*nparam.(fld) + talpha*oparam.(fld);
end

end

function [kl] = gaussian_diverge(param1, param2)
s = param2.omega/param1.omega;
dmu = param2.omega\param2.eta - param1.omega\param1.eta;
kl = 0.5*(dmu'*param2.omega*dmu - numel(dmu) + trace(s) - log(max(0,det(s))));

end

function [param] = gaussian_smpls2param(smpls, ~)
[nsmpls, d] = size(smpls);

tsigma = cov(smpls, 0);
param.omega = ((nsmpls-d-2)/(nsmpls-1)).*spinv(tsigma);

tmu = mean(smpls, 1)';
param.eta = param.omega*tmu;

end

function Q = spinv(P)
Q = pinv(P);
Q = 0.5*(Q + Q');
end

function [smplmean] = gaussian_param2smplm(param)
smplmean = (param.omega\param.eta)';
end

function [err] = gaussian_errmsr(tsmpls, smpls, param)
tsigma = cov(tsmpls, 0); tmu = mean(tsmpls, 1)';
ssigma = cov(smpls, 0); smu = mean(smpls, 1)';
psigma = spinv(param.omega); pmu = (param.omega\param.eta);

err.mu_mse = mean((pmu - tmu).^2);
err.sigma_mse = (norm(psigma - tsigma, 'fro')./numel(tmu)).^2;
err.sigma_msle = (norm(log(psigma./tsigma), 'fro')./numel(tmu)).^2;
s = psigma\tsigma;
err.kl = 0.5*((pmu-tmu)'*(psigma\(pmu-tmu)) - numel(tmu) + trace(s) - log(max(0,det(s))));

err.s_mu_mse = mean((smu - tmu).^2);
err.s_sigma_mse = (norm(ssigma - tsigma, 'fro')./numel(tmu)).^2;
err.s_sigma_msle = (norm(log(ssigma./tsigma), 'fro')./numel(tmu)).^2;
s = ssigma\tsigma;
err.s_kl = 0.5*((smu-tmu)'*(ssigma\(smu-tmu)) - numel(tmu) + trace(s) - log(max(0,det(s))));

end


function logpdf = gaussian_qklogpdf(smpl, param)
logpdf = -0.5*(smpl*param.omega*smpl' - 2*smpl*param.eta);

end

function glogpdf = gaussian_qkgradlogpdf(smpl, param)
glogpdf = param.eta - param.omega*smpl';

end
