% Generate synthetic data for the model.
% 
% Minjie Xu (chokkyvista06@gmail.com)

function [data, model] = sparsebayeslogreg_gendata(n, d, model)
assert(strcmp([model.id, '_gendata'], mfilename));

assert(model.dim == d);
mu = model.param.mu;
b = model.param.b;

P = rand(d);
X = mvnrnd(rand(1,d), P*P', n)'; % X is D x N matrix
w = rand(d,1)-0.5; % w is D x 1 vector
w = mu - b.*sign(w).*log(1-2*abs(w));
model.param.w = w;
y = (1./(1+exp(-w'*X)) >= rand(1,n))*2 - 1; % y is 1 x N vector consisting of +/- 1 entries
data = struct('x', num2cell(X,1), 'y', num2cell(y), 'yx', num2cell(bsxfun(@times,X,y),1));

end
