function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

% init values that need cross checking one by one on all combinations
C_sig_values = [0.01,0.03,0.1,0.3,1,3,10,30]
min_error = inf % start value should be high

% for every value of c try every value of sigma
for c = C_sig_values
  for sig = C_sig_values
    % Train the model:
    % 'x1' and 'x2' are dummy parameters. They are filled-in at runtime when 
    % svmTrain() calls your kernel function. This is done by: @(x1, x2)
    model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));
    % Compute predictions:
    svmPredict(model, Xval);
    % Compute the error between predictions and yval.  Error is deﬁned as the 
    % fraction of the cross validation examples that were classiﬁed incorrectly. 
    % In Octave/MATLAB, you can compute this error using:
    % mean(double(predictions ~= yval))
    err   = mean(double(svmPredict(model, Xval) ~= yval));
    % WHen the error is lower then the stored error then save sigma and C asctime
    % they result in the best known error
    if( err <= min_error )
      C = c;
      sigma = sig;
      min_error = err;
      fprintf('Minimum found C = %f, sigma = %f, error = %f', C, sigma, min_error)
    end
  end;
end;





% =========================================================================

end
