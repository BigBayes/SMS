% The model for Sparse Bayesian Logistic Regression (with Laplace prior).
% For details, please refer to comments in 'smssample'.
% 
% Minjie Xu (chokkyvista06@gmail.com)

function model = sparsebayeslogreg(d, mu, b)
if nargin < 3
    mu = zeros(d, 1);
    b = ones(d, 1);
end
mu = mu.*ones(d, 1);
b = b.*ones(d, 1);
    
model.id = 'sparsebayeslogreg';
model.dim = d;
model.param = struct('mu', mu, 'b', b);

model.prior = @(w)prod(exp(-abs(w-mu)./b));
model.logprior = @(w)sum(-abs(w-mu))./b;
model.gradlogprior = @(w)sign(mu-w)./b;

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
postpred = zeros(n, 1);
for i = 1:n
    postpred(i) = mean(1./(1+exp(-(ws*x(:,i)))));
end
stats = [stats; postpred];

end