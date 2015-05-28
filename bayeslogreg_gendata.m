% Generate synthetic data for the model.
% 
% Minjie Xu (chokkyvista06@gmail.com)

function [data, model] = bayeslogreg_gendata(n, d, model)
assert(strcmp([model.id, '_gendata'], mfilename));

if exist('model', 'var')
    assert(model.dim == d);
    mu = model.param.mu;
    sigma = model.param.sigma;
else
    P = rand(d);
    mu = rand(d, 1);
    sigma = P'*P;
end

P = rand(d);
X = mvnrnd(rand(1,d), P*P', n)'; % X is D x N matrix
w = mvnrnd(mu', sigma)'; % w is D x 1 vector
model.param.w = w;
% y = (w'*X >= 0)*2 - 1;
y = (1./(1+exp(-w'*X)) >= rand(1,n))*2 - 1; % y is 1 x N vector consisting of +/- 1 entries
data = struct('x', num2cell(X,1), 'y', num2cell(y), 'yx', num2cell(bsxfun(@times,X,y),1));

% [~,idx] = sort(rand(1,d)*X);
% data = data(idx);

end

% x = [data.x];
% xx = bsxfun(@minus,x,mean(x,2));
% coeff = pca(xx');
% coeff = coeff(:,1:2);
% y = [data.y];
% tdnormvis(coeff'*model.param.mu, coeff'*model.param.sigma*coeff, 3, 15, 'LineWidth', 2);
% hold on;
% scatter(coeff(:,1)'*x(:,y>0), coeff(:,2)'*x(:,y>0), 'ro', 'LineWidth', 1.5);
% hold on;
% scatter(coeff(:,1)'*x(:,y<0), coeff(:,2)'*x(:,y<0), 'bo', 'LineWidth', 1.5);
% legend({'p_0(w)', 'y_i = +1', 'y_i = - 1'}, 'interpreter', 'tex');
% title('synthetic dataset and model prior (projected)');
% xlabel('principal component #1');
% ylabel('principal component #2');
