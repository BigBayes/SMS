% The model for Bayesian Logistic Regression.
% For details, please refer to comments in 'smssample'.
% 
% Minjie Xu (chokkyvista06@gmail.com)

function model = bayeslogreg(d, mu, sigma)
if nargin < 3
    mu = zeros(d, 1);
    sigma = eye(d);
end
model.id = 'bayeslogreg';
model.dim = d;
model.param = struct('mu', mu, 'sigma', sigma);

Q = chol(sigma);
Qt = Q';
model.prior = @(w)exp(-0.5*sum((Qt\(w-mu)).^2));
model.logprior = @(w)-0.5*sum((Qt\(w-mu)).^2);
model.gradlogprior = @(w)Q\(Qt\(mu-w));

model.lklhd = @(datum, w)1./(1+exp(-(w'*datum.yx)));
model.loglklhd = @(datum, w)-log(1+exp(-(w'*datum.yx)));
model.gradloglklhd = @(datum, w)datum.yx./(1+exp(w'*datum.yx));

model.data2stats = @(data)[data.yx];
model.statlklhd = @statlklhd;
model.logstatlklhd = @logstatlklhd;
model.gradlogstatlklhd = @gradlogstatlklhd;

model.statfun = @(ws, data)statfun(ws, data);

end

function p = statlklhd(yx, w)
p = 1./prod(1+exp(-(w'*yx)));
end

function p = logstatlklhd(yx, w)
p = -sum(log(1+exp(-(w'*yx))));
end

function p = gradlogstatlklhd(yx, w)
p = sum(bsxfun(@rdivide, yx, 1+exp(w'*yx)), 2);
end

function stats = statfun(ws, data)
stats = [];
x = [data.x];
n = numel(data);
% postpred = mean(1./(1+exp(-(ws*x))), 1)';
postpred = zeros(n, 1);
for i = 1:n
    postpred(i) = mean(1./(1+exp(-(ws*x(:,i)))));
end
stats = [stats; postpred];

end