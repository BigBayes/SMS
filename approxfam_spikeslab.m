% Uses paired factorization approximation for spikeslab prior with Gaussian
% likelihood
%
% For details, please refer to comments in 'smssample'.
% 
% Minjie Xu (chokkyvista06@gmail.com) & 
% Balaji Lakshminarayanan (balaji@gatsby.ucl.ac.uk)

function spikeslabaf = approxfam_spikeslab()
% required fields
spikeslabaf.id = 'spikeslab';
spikeslabaf.validparam = @spikeslab_validparam;
spikeslabaf.emptyparam = @spikeslab_emptyparam;
spikeslabaf.priorparam = @spikeslab_priorparam;
spikeslabaf.mulall = @spikeslab_mulall;
spikeslabaf.mul = @spikeslab_mul;
spikeslabaf.div = @spikeslab_div;
spikeslabaf.damp = @spikeslab_damp;
spikeslabaf.smpls2param = @spikeslab_smpls2param;
spikeslabaf.param2smplm = @spikeslab_param2smplm;
spikeslabaf.diverge = @spikeslab_diverge;
spikeslabaf.errmsr = @spikeslab_errmsr;
end


function isvalid = spikeslab_validparam(param)
if ~all(isfield(param, {'invsigma', 'invsigmamu', 'logodds'}))
    isvalid = false;
    return;
end
isvalid = all(param.invsigma >= 0);
end


function [param] = spikeslab_emptyparam(model)
param.invsigma = zeros(model.dim, 1);
param.invsigmamu = zeros(model.dim, 1);
param.logodds = zeros(model.dim, 1);
end


function [param, exact] = spikeslab_priorparam(model, m)
if nargin == 1
    m = 1;
end
switch model.id
    case {'spikeslab'}
        % p(var|param) is an APPROXIMATION to p(var|prior_param)^{1/m}
        param.invsigma = ones(model.dim, 1) ./ model.param.wvariance ./ m;
        param.invsigmamu = model.param.mu ./ model.param.wvariance ./ m;
        param.logodds = ones(model.dim, 1) * log(model.param.pi./m) - log(1 - model.param.pi./m);
        exact = true; % when m=1, p(var|param) = p(var|prior_param)
    otherwise
        error('approxfam ''spikeslab'' does not support model ''%s''!', ...
            model.id);
end
end


function [param] = spikeslab_mulall(params)
param = params(1);
flds = fieldnames(param);
for i = 2:numel(params)
    for j = 1:numel(flds)
        fld = flds{j};
        param.(fld) = param.(fld) + params(i).(fld);
    end
end
end


function [param] = spikeslab_mul(param1, param2)
param = param1;
flds = fieldnames(param);
for i = 1:numel(flds)
    fld = flds{i};
    param.(fld) = param1.(fld) + param2.(fld);
end
end


function [param] = spikeslab_div(param1, param2)
param = param1;
flds = fieldnames(param);
for i = 1:numel(flds)
    fld = flds{i};
    param.(fld) = param1.(fld) - param2.(fld);
end
end


function [param] = spikeslab_damp(nparam, oparam, dampalpha)
% assuming dampalpha holding the same fields as param
param = nparam;
flds = fieldnames(param);
for i = 1:numel(flds)
    fld = flds{i};
    % always damp 'logodds' with damping factor 0.5
    if strcmpi(fld, 'logodds')
        param.(fld) = 0.5*nparam.(fld) + 0.5*oparam.(fld);
        continue;
    end
    if isstruct(dampalpha)
        talpha = dampalpha.(fld);
    else
        talpha = dampalpha;
    end
    param.(fld) = (1-talpha)*nparam.(fld) + talpha*oparam.(fld);
end
end


function kl = spikeslab_diverge(param1, param2)
mu1 = param1.invsigmamu ./ param1.invsigma;
mu2 = param2.invsigmamu ./ param2.invsigma;
dmu = mu1 - mu2;
kl_gauss = 0.5 * (log(param1.invsigma) - log(param2.invsigma)) - 1 + ...
          param2.invsigma ./ param1.invsigma + (dmu.^2) .* param2.invsigma;
kl_bernoulli = sigmoid(param1.logodds) .* (param1.logodds - param2.logodds) ...
                + log(sigmoid(-param1.logodds)) - log(sigmoid(-param2.logodds));
bernoulli_mean1 = sigmoid(param1.logodds);
kl = sum(kl_bernoulli) + sum(bernoulli_mean1 .* kl_gauss);
end


function [param] = spikeslab_smpls2param(samples, defaultparam)
% it is important to define param in the same order as in parent code
% (invsigma, invsigmamu, logodds)
% otherwise, matlab seems to generate error during assignment
W = reshape([samples.W], [], size(samples,1))';
S = reshape([samples.S], [], size(samples,1))';
[n_samples, d] = size(W);
n_samples_s1 = sum(S, 1);

dirichlet_concentration = 1e-3; % Dirichlet concentration
base_distribution = 0.5*ones(2, d); % uniform prior
assert(all(sum(base_distribution, 1) == 1)); % every column needs to sum to 1
q_alpha = (n_samples_s1 + dirichlet_concentration * base_distribution(2,:)) ...
            ./ (n_samples + dirichlet_concentration);

q_sw = sum(S .* W, 1);
q_sw2 = sum(S .* (W.^2), 1); % sum of w^2

q_var =  (q_sw2 - (q_sw.^2)./n_samples_s1) ./  (n_samples_s1 - 1);
param.invsigma = (((n_samples_s1 - 3) / (n_samples_s1 - 1)) ./ q_var)';
param.invsigmamu = (q_sw ./ n_samples_s1)' .* param.invsigma;

% set posterior over w to prior if very few samples for s=1
n_threshold = 3; % chose 3 due to n-3 factor in precision formula above
idx_few_samples_s1 = n_samples_s1' <= n_threshold;

param.invsigma(idx_few_samples_s1) = defaultparam.invsigma(idx_few_samples_s1);
param.invsigmamu(idx_few_samples_s1) = defaultparam.invsigmamu(idx_few_samples_s1);

param.logodds = (log(q_alpha) - log(1-q_alpha))';

assert(all(param.invsigma >= 0));
end


function [smplmean] = spikeslab_param2smplm(param)
smplmean.W = (ones(size(param.invsigmamu)).*sigmoid(param.logodds))';
smplmean.S = (param.invsigmamu ./ param.invsigma)';
end


function [err] = spikeslab_errmsr(tsmpls, smpls, param)
err.kl = nan;
err.mse = nan;
end
