% The diagonal multivariate-Gaussian approximating family.
% For details, please refer to comments in 'smssample'.
% 
% Minjie Xu (chokkyvista06@gmail.com)

function diagaussaf = approxfam_diagauss()
% required fields
diagaussaf.id = 'diagauss';
diagaussaf.validparam = @diagauss_validparam;
diagaussaf.emptyparam = @diagauss_emptyparam;
diagaussaf.priorparam = @diagauss_priorparam;
diagaussaf.mulall = @diagauss_mulall;
diagaussaf.mul = @diagauss_mul;
diagaussaf.div = @diagauss_div;
diagaussaf.damp = @diagauss_damp;
diagaussaf.diverge = @diagauss_diverge;
diagaussaf.smpls2param = @diagauss_smpls2param;
diagaussaf.param2smplm = @diagauss_param2smplm;
diagaussaf.errmsr = @diagauss_errmsr;

% optional fields
diagaussaf.qklogpdf = @diagauss_qklogpdf;
diagaussaf.qkgradlogpdf = @diagauss_qkgradlogpdf;

end


function isvalid = diagauss_validparam(param)
if ~all(isfield(param, {'eta', 'omega'}))
    isvalid = false;
    return;
end
isvalid = all(param.omega >= 0);

end

function [param] = diagauss_emptyparam(model)
param.omega = zeros(model.dim, 1);
param.eta = zeros(model.dim, 1);

end

function [param, exact] = diagauss_priorparam(model, m)
if nargin == 1
    m = 1;
end
switch model.id
    case {'bayeslogreg', 'diagauss'}
        param.omega = (1/m)./diag(model.param.sigma);
        param.eta = param.omega.*model.param.mu;
        exact = false;
    case {'sparsebayeslogreg'}
        param.omega = 0.5./(model.param.b.^2.*m);
        param.eta = param.omega.*model.param.mu;
        exact = false;
    otherwise
        error('approxfam ''diagauss'' does not yet support model ''%s''!', ...
            model.id);
end

end

function [param] = diagauss_mulall(params)
param = params(1);
flds = fieldnames(param);
for i = 2:numel(params)
    for j = 1:numel(flds)
        fld = flds{j};
        param.(fld) = param.(fld) + params(i).(fld);
    end
end

end

function [param] = diagauss_mul(param1, param2)
param = param1;
flds = fieldnames(param);
for i = 1:numel(flds)
    fld = flds{i};
    param.(fld) = param1.(fld) + param2.(fld);
end

end

function [param] = diagauss_div(param1, param2)
param = param1;
flds = fieldnames(param);
for i = 1:numel(flds)
    fld = flds{i};
    param.(fld) = param1.(fld) - param2.(fld);
end

end

function [param] = diagauss_damp(nparam, oparam, dampalpha)
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

function [kl] = diagauss_diverge(param1, param2)
s = param2.omega./param1.omega;
dmu = param2.eta./param2.omega - param1.eta./param1.omega;
kl = 0.5*(param2.omega'*(dmu.^2) - numel(dmu) + sum(s) - log(max(0,prod(s))));

end

function [param] = diagauss_smpls2param(smpls, ~)
param.omega = 1./var(smpls, 0)';

tmu = mean(smpls, 1)';
param.eta = param.omega.*tmu;

end

function [smplmean] = diagauss_param2smplm(param)
smplmean = (param.eta./param.omega)';
end

function [err] = diagauss_errmsr(tsmpls, smpls, param)
tsigma = var(tsmpls, 0)'; tmu = mean(tsmpls, 1)';
ssigma = var(smpls, 0)'; smu = mean(smpls, 1)';
psigma = 1./param.omega; pmu = param.eta./param.omega;

err.mu_mse = mean((pmu - tmu).^2);
err.sigma_mse = mean((psigma - tsigma).^2);
err.sigma_msle = mean((log(psigma./tsigma)).^2);
s = tsigma./psigma;
err.kl = 0.5*(sum((pmu-tmu).^2./psigma) - numel(tmu) + sum(s) - log(max(0,prod(s))));

err.s_mu_mse = mean((smu - tmu).^2);
err.s_sigma_mse = mean((ssigma - tsigma).^2);
err.s_sigma_msle =  mean((log(ssigma./tsigma)).^2);
s = tsigma./ssigma;
err.s_kl = 0.5*(sum((smu-tmu).^2./ssigma) - numel(tmu) + sum(s) - log(max(0,prod(s))));

end


function logpdf = diagauss_qklogpdf(smpl, param)
logpdf = -0.5*((smpl.^2)*param.omega - 2*smpl*param.eta);

end

function glogpdf = diagauss_qkgradlogpdf(smpl, param)
glogpdf = param.eta - param.omega.*smpl';

end
