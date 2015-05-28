% The Linear Gaussian model.
% For details, please refer to comments in 'smssample'.
% 
% Minjie Xu (chokkyvista06@gmail.com)

function [model] = gaussian(d, mu, sigma, lsigma)
if nargin < 4
    mu = zeros(d, 1);
    sigma = eye(d);
    lsigma = eye(d);
end
model.id = 'gaussian';
model.dim = d;
model.param = struct('mu', mu, 'sigma', sigma, 'lsigma', lsigma, ...
    'pmu', mu, 'psigma', sigma);

Q = chol(sigma);
Qt = Q';
model.prior = @(w)exp(-0.5*sum((Qt\(w-mu)).^2));
model.logprior = @(w)-0.5*sum((Qt\(w-mu)).^2);
model.gradlogprior = @(w)Q\(Qt\(mu-w));

R = chol(lsigma);
Rt = R';
model.lklhd = @(datum, w)exp(-0.5*sum((Rt\(w-datum.x)).^2));
model.loglklhd = @(datum, w)-0.5*sum((Rt\(w-datum.x)).^2);
model.gradloglklhd = @(datum, w)R\(Rt\(datum.x-w));

model.data2stats = @(data)struct('xbar',mean([data.x],2),'n',numel(data));
model.statlklhd = @(ss, w)exp(-0.5*ss.n*sum((Rt\(ss.xbar-w)).^2));
model.logstatlklhd = @(ss, w)-0.5*ss.n*sum((Rt\(ss.xbar-w)).^2);
model.gradlogstatlklhd = @(ss, w)ss.n*(R\(Rt\(ss.xbar-w)));

end

