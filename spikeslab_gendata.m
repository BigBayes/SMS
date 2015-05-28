% Generate synthetic data for the model.
function [x, y] = spikeslab_gendata(n, model)

assert(strcmp([model.id, '_gendata'], mfilename));

P = rand(model.dim);
x = mvnrnd(rand(1,model.dim), P*P', n)'; % X is d x n matrix
noise = randn(1,n) * sqrt(model.param.noisevariance);
y = model.param.w'*x + noise; % y is 1 x n vector

end
