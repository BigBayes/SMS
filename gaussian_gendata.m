% Generate synthetic data for the model.
% 
% Minjie Xu (chokkyvista06@gmail.com)

function [data] = gaussian_gendata(n, d, model)
assert(strcmp([model.id, '_gendata'], mfilename));

if exist('model', 'var')
    assert(model.dim == d);
    mu = model.param.mu;
    sigma = model.param.sigma;
    lsigma = model.param.lsigma;
else
    P = rand(d);
    lsigma = P'*P;
    mu = zeros(d,1);
    sigma = eye(d);
end

X = mvnrnd(mvnrnd(mu',sigma), lsigma, n)';
data = struct('x', num2cell(X,1));

% [~,idx] = sort(rand(1,d)*X);
% data = data(idx);

end