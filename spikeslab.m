% The model for sparse linear regression using spike and slab priors
%
% Inputs:
% d is the number of dimensions
% mu is a d x 1 vector
% wvariance is a scalar denoting variance of wtilde
% pi is a scalar denoting probability of s being non-zero
% noisevariance is the assumed variance of noise e = y - wx

function model = spikeslab(d, mu, wvariance, pi, noisevariance)
model.id = mfilename;
model.dim = d;
model.param = struct('mu', mu,  'wvariance', wvariance, ...
    'pi', pi, 'noisevariance', noisevariance);

model.param.wtilde = mu + randn(d, 1)*sqrt(wvariance); % w is d x 1 vector
model.param.s = rand(d, 1) <= pi;
model.param.w = model.param.wtilde .* model.param.s;

model.logprior = @(var) -0.5*log(wvariance) ...
    -0.5*sum((var.wtilde-mu).^2)/wvariance ...
    + sum(var.s)*log(pi) + sum(1-var.s)*log(1-pi);
model.prior = @(var) exp(model.logprior(var));

model.loglklhd = @(datum, var) -0.5*log(noisevariance) ...
    -(0.5/noisevariance)*((datum.y-var.w'*datum.x).^2) ;
model.lklhd = @(datum, var) exp(model.loglklhd(datum, var));

model.data2stats = @(data) struct('XtX',[data.x]*[data.x]','XtY',[data.x]*[data.y]');
model.logstatlklhd = @(ss, var) -0.5*log(noisevariance) ...
    -(0.5/noisevariance)*(-2*var.w'*ss.XtY + var.w'*ss.XtX*var.w);
model.statlklhd = @(ss, var) exp(model.logstatlklhd(ss, var));
end