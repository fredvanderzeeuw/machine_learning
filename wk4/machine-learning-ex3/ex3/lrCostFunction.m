function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 

% Predictions of all examples stored in h
h = sigmoid(X * theta); 

% Cost function logistic regression = cost + regularization
% Calculate regularization
regLambda = ( theta' * theta * lambda ) / ( 2 * m ) - ( theta(1) * theta(1) * lambda ) / ( 2 * m );
% Add Ccost + regularization
J = 1/m * (-y' * log(h) - (1 - y)' * log(1 - h)) + regLambda;

% Create a mask for later use as part of the following formula needs theta_0 to
% be excluded ... this is done by multiplying by ones except 1 zero for theta_0
mask = ones(size(theta)); mask(1) = 0;

% Calc the gradient for log regress cost function with regularization and using 
% mask to exclude theta_0 from the last part of the formula
grad = 1/m * X' * (sigmoid(X * theta) - y) + lambda * (theta .* mask)/ m;

% old formula without regularization
% grad = 1/m * X' * (h-y)

% =============================================================

% Transpose gradient
grad = grad(:);

end
