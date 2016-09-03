function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% calculate cost function
%cost = X*theta - y;
% regularization part with only theta 1! So exclude theta0 as we do not want to
% regularize X0 as it is the input
%theta1 = [0 ; theta(2:end, :)];
%J = 1/(2*m) * (cost'*cost) + 1/(2*m) * lambda*(theta1'*theta1);

% calculate gradients
%grad = (X' * cost + lambda * theta1) / m;


h = X*theta;

% regularize theta by removing first value
theta_reg = [0;theta(2:end, :);];
J = (1/(2*m))*sum((h-y).^2)+(lambda/(2*m))*theta_reg'*theta_reg;

grad = (1/m)*(X'*(h-y)+lambda*theta_reg);




% =========================================================================

grad = grad(:);

end
