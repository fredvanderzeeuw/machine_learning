function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% m = number of rows
m = size(X, 1);
% num_labels = number of labels
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to X in first column for theta_0, , [A B] just pasts matrix B
% after matrix A 
a_1 = [ones(m,1) X];
% Calculate a_2 matrix outputs in two steps: first z value then sigmoid(z)
z_2 = a_1 * Theta1';
% a_2 = first column of ones followed by sigmoid (z)
a_2 = [ones(size(z_2),1) sigmoid(z_2)];
% With output a_2 calculate output a_3 again in two steps with z and sigmoid(z)
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);
% Use the max function to identify the highest p-values and their index as the
% most likely label to be predicted
[p_max, index_max] = max(a_3, [], 2);
% return the index as the matrix with labels
p = index_max;



% =========================================================================


end
