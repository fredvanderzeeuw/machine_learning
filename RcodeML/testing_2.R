################################################################################
# ML copy of class method follow along ex4.m octave code of the ML course of   #
# Coursera Stanford Andrew Ng                                                  #
# Subject: ML with neural network                                              #
################################################################################
#  Machine Learning Online Class - Exercise 4 Neural Network Learning
# 
#   Instructions
#   ------------
#      
#   This file contains code that helps you get started on the
#   linear exercise. You will need to complete the following functions 
#   in this exericse:
#     
#      sigmoidGradient.m
#      randInitializeWeights.m
#      nnCostFunction.m
# 
#   For this exercise, you will not need to change any code in this file,
#   or any other files other than those mentioned above.

# initialize 
# octave code: clear ; close all; clc
rm(list = ls())
# load library to read matlab data files
library(R.matlab)


################################################################################
# sigmoid function                                                             #
# use = sigmoid(matrix(rnorm(10),nrow = 5, ncol = 2))                          #
# output = matrix with element wise sigmoid value                              #
# test: sigmoid(matrix(data = c(1, -0.5, 0, 0.5, 1),nrow = 5,ncol = 1))        #
# test output:                                                                 #
# [1,] 0.7310586                                                               #
# [2,] 0.3775407                                                               #
# [3,] 0.5000000                                                               #
# [4,] 0.6224593                                                               #
# [5,] 0.7310586                                                               #
################################################################################
sigmoid <- function(z){
    # Compute sigmoid function octave coed: g =  1./(1+exp(-z))
    # J = sigmoid(z) computes the sigmoid of z.
    return(1 / (1 + exp(-z)))
}


################################################################################
# sigmoidGradient function                                                     #
# use = sigmoid(matrix(rnorm(10),nrow = 5, ncol = 2))                          #
# output = matrix with element wise sigmoid value                              #
# test: sigmoidGradient(matrix(data = c(1, -0.5, 0, 0.5, 1),nrow = 5,ncol = 1))#
# test output:                                                                 #
# [1,] 0.1966119                                                               #
# [2,] 0.2350037                                                               #
# [3,] 0.2500000                                                               #
# [4,] 0.2350037                                                               #
# [5,] 0.1966119                                                               #
################################################################################
sigmoidGradient <- function(z){
    # Octave code: g = sigmoid(z) .* (1-sigmoid(z));
    return(sigmoid(z) * (1 - sigmoid(z)))
}


################################################################################
# randInitializeWeights function                                               #
# use = randInitializeWeights((L_in, L_out))                                   #
# initial_Theta1 <- randInitializeWeights(input_layer_size, hidden_layer_size) # 
# initial_Theta2 <- randInitializeWeights(hidden_layer_size, num_labels)       #
# test: randInitializeWeights(2, 3)                                            #
# test output:                                                                 #
# [,1]         [,2]        [,3]                                                #
# [1,]  0.03981075 -0.081502653  0.05205921                                    #
# [2,] -0.06962542 -0.005159448  0.02886089                                    #
# [3,] -0.06152138 -0.061020188 -0.11389310                                    #
################################################################################
randInitializeWeights <- function(L_in, L_out){
    #RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
    #incoming connections and L_out outgoing connections
    #   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
    #   of a layer with L_in incoming connections and L_out outgoing 
    #   connections. 
    #
    #   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
    #   the column row of W handles the "bias" terms
    #
    
    # You need to return the following variables correctly 
    # octave code: W = zeros(L_out, 1 + L_in);
    W <- matrix(0, nrow = L_out, ncol = L_in + 1)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Initialize W randomly so that we break the symmetry while
    #               training the neural network.
    # Note: The first row of W corresponds to the parameters for the bias units
    
    # octave code: epsilon_init = 0.12;
    epsilon_init <- 0.12
    # octave code: W = rand(L out, 1 + L in) * 2 * epsilon init ??? epsilon init;
    W <- matrix(runif(L_out*(1+L_in)), L_out,(L_in + 1)) * 2 * epsilon_init - epsilon_init
    
    return(W)
    # =========================================================================
}


################################################################################
# debugInitializeWeights function                                              #
# use = debugInitializeWeights(fan_out, fan_in)                                #
# test: debugInitializeWeights(2,3)                                            #
# test output:                                                                 #
# [,1]       [,2]       [,3]      [,4]                                         #
# [1,] 0.8414710  0.1411200 -0.9589243 0.6569866                               #
# [2,] 0.9092974 -0.7568025 -0.2794155 0.9893582                               #
################################################################################
debugInitializeWeights <- function(fan_out, fan_in){
    #DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
    #incoming connections and fan_out outgoing connections using a fixed
    #strategy, this will help you later in debugging
    #   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
    #   of a layer with fan_in incoming connections and fan_out outgoing 
    #   connections using a fix set of values
    #
    #   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
    #   the first row of W handles the "bias" terms
    
    # Set W to zeros
    # octave code: W = zeros(fan_out, 1 + fan_in);
    W <- matrix(0, fan_out, (1+fan_in))
    
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    # octave code: W = reshape(sin(1:numel(W)), size(W)) / 10;
    W <- matrix(sin(1:(nrow(W) * ncol(W))), fan_out, (1+fan_in))
    
    return(W)
}

################################################################################
# numgrad function                                                             #
# use = computeNumericalGradient(nnCostFunction, nn_params,input_layer_size,   #
# hidden_layer_size, num_labels, X, y, lambda)                                 #
# Is used to compute an estimate of the gradient to see whether gradients      #
# calculated in backprop are reasonable.                                       #
# output = a 1 dimensional matrix with computed gradients which can be compa-  #
# red to the gradients computed with the cost function.                        #
################################################################################
computeNumericalGradient <- function(J, theta,input_layer_size, 
                                     hidden_layer_size, num_labels, X, y, lambda){
    #COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    #and gives us a numerical estimate of the gradient.
    #   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
    #   gradient of the function J around theta. Calling y = J(theta) should
    #   return the function value at theta.
    
    # Notes: The following code implements numerical gradient checking, and 
    #        returns the numerical gradient.It sets numgrad(i) to (a numerical 
    #        approximation of) the partial derivative of J with respect to the 
    #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
    #        be the (approximately) the partial derivative of J with respect 
    #        to theta(i).)
    # octave code: numgrad = zeros(size(theta));
    numgrad <- matrix(0, length(theta), 1)
    # octave code: perturb = zeros(size(theta));
    perturb <- matrix(0, length(theta), 1)
    e <- 1e-4
    
    # loop over elements of p
    # octave code: for p = 1:numel(theta)
    for(p in 1:length(theta)){
        # Set perturbation vector
        perturb[p] <- e
        loss1 <- nnCostFunction((theta - perturb),input_layer_size, hidden_layer_size, 
                                num_labels, X, y, lambda)[[1]]
        loss2 <- nnCostFunction((theta + perturb),input_layer_size, hidden_layer_size, 
                       num_labels, X, y, lambda)[[1]]
        # Compute Numerical Gradient
        numgrad[p] <- (loss2 - loss1) / (2*e)
        perturb[p] <- 0
    }
    return(numgrad)
}
   

################################################################################
# checkNNGradients function                                                    #
# use = checkNNGradients(lambda)                                               #
# Is used to check whether the function nnCostFunction returns the right grad- #
# ients based on a small neural network that is generated within the running   #
# checkNNGradients function.                                                   #
# output = a printed table with gradients form the nnCostFunction and gradients#
# that where based on estimation.                                              #
################################################################################
checkNNGradients <- function(lambda){
    #CHECKNNGRADIENTS Creates a small neural network to check the
    #backpropagation gradients
    #   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
    #   backpropagation gradients, it will output the analytical gradients
    #   produced by your backprop code and the numerical gradients (computed
    #   using computeNumericalGradient). These two gradient computations should
    #   result in very similar values.
    
    # octave code: input_layer_size = 3;
    input_layer_size <- 3
    # octave code: hidden_layer_size = 5;
    hidden_layer_size <- 5
    # octave code: num_labels = 3;
    num_labels <- 3
    # octave code: m = 5;
    m <- 5
    
    # We generate some 'random' test data
    # octave code: Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
    Theta1 <- debugInitializeWeights(hidden_layer_size, input_layer_size)
    # octave code: Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
    Theta2 <- debugInitializeWeights(num_labels, hidden_layer_size)
    
    # Reusing debugInitializeWeights to generate X
    # octave code: X  = debugInitializeWeights(m, input_layer_size - 1);
    X  <- debugInitializeWeights(m, (input_layer_size - 1))
    
    # octave code: y  = 1 + mod(1:m, num_labels)';
    y <- matrix(1 + (1:m %% num_labels),m,1)
    
    # Unroll parameters
    # octave code: nn_params = [Theta1(:) ; Theta2(:)];
    nn_params <- matrix(c(c(Theta1), c(Theta2)),(length(Theta1)+length(Theta2)),1)
    
    # Short hand for cost function
    # octave code: costFunc = @(p) nnCostFunction(p, input_layer_size, 
    # hidden_layer_size, num_labels, X, y, lambda);
    # no shorthand used in R!
    cost_grad <- nnCostFunction(nn_params,input_layer_size, hidden_layer_size, 
                                num_labels, X, y, lambda)
    # octave code: [cost, grad] = costFunc(nn_params);
    cost <- cost_grad[[1]]
    grad <- cost_grad[[2]]
    
    # octave code: numgrad = computeNumericalGradient(costFunc, nn_params);
    numgrad <- computeNumericalGradient(nnCostFunction, nn_params,input_layer_size, 
                                        hidden_layer_size, num_labels, X, y, lambda)
    
    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar. 
    # octave code: disp([numgrad grad]);
    print(cbind(numgrad, grad, numgrad - grad))
    print("The 1st column is the computed gradient, 2nd is analytical, 3rd is the difference.")
    print("The difference should be less than 1e-9")
    
    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    # octave code: diff = norm(numgrad-grad)/norm(numgrad+grad);
    difference <- norm(numgrad - grad, type = "2") / norm(numgrad + grad, type = "2")
    
    print("If your backpropagation implementation is correct, then the relative difference will be small")
    print("less than 1e-9")
    print(difference)
}


################################################################################
# display_data function                                                        #
# use = display_data(sel,row_nr = 20, col_nr = 20)                             #
# output = one matrix with of 10 by 10 matrices of images of 20 by 20 pixels   #
# test: image(display_data(sel, 20,20))                                        #
# test output: one image of 400 by 400 pixels                                  #
################################################################################
display_data <- function(matrix_data, row_nr ,col_nr){
    # function displays handwriting data in a 10 x 10 matrix image
    
    # init final image matrix that contains all the images
    img_all <- matrix(data = NA, nrow = 10 * row_nr, ncol = 10 * col_nr)
    # loop through matrix and get images one by one 
    for(i in 1:100){
        # get image
        img <- matrix(sel[i,], nrow = row_nr, ncol = col_nr)
        # get column and row where to place in final image
        img_all_col <- (((i- ((ceiling(i/10)-1)*10))-1) * 20)+1
        img_all_row <- (ceiling(i/10)*20 - 19)
        # place image in the final matrix
        img_all[img_all_row:(img_all_row + row_nr-1),img_all_col:(img_all_col + col_nr-1)] <- img[1:20,1:20]
    }
    # R plot rotates 90 degrees CW so t(apply(img_all, 2, rev)) rotates back
    return(t(apply(img_all, 2, rev)))
}

################################################################################
# nnCostFunction function                                                      #
# use = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,         #
# num_labels, X, y, lambda)                                                    #
# output = a list with the cost (one value) and the gradients which is an      #
# unrolled vector of gradients for Theta1 and Theta2                           #
# test: J <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size,    #
# num_labels, X, y, lambda)[[1]]                                               #
# test: grad <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size, #
# num_labels, X, y, lambda)[[2]]                                               #
# test output: the cost J selected from the output-list J,grad                 #
# test output: the gradient grad selected from the output list J,grad          #
################################################################################
nnCostFunction <- function(nn_params, input_layer_size, hidden_layer_size, 
                           num_labels, X, y, lambda){
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #    X, y, lambda) computes the cost and gradient of the neural network. The
    #    parameters for the neural network are "unrolled" into the vector
    #    nn_params and need to be converted back into the weight matrices. 
    #  
    #    The returned parameter grad should be a "unrolled" vector of the
    #    partial derivatives of the neural network.
    # 
    # 
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    # octave code: Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    #                  hidden_layer_size, (input_layer_size + 1));
    Theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
                     hidden_layer_size,(input_layer_size + 1))
    # octave code: Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    #                  num_labels, (hidden_layer_size + 1));
    Theta2 <- matrix(nn_params[(1+(hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
                     num_labels,(hidden_layer_size + 1))
    
    #  Setup some useful variables
    m <- nrow(X)
    # 
    #  You need to return the following variables correctly 
    J <- 0
    # octave code: Theta1_grad = zeros(size(Theta1));
    Theta1_grad <- matrix(0,nrow(Theta1), ncol(Theta1))
    # octave code: Theta2_grad = zeros(size(Theta2));
    Theta2_grad <- matrix(0,nrow(Theta2), ncol(Theta2))
    
    #  ====================== YOUR CODE HERE ======================
    #      Instructions: You should complete the code by working through the
    #                following parts.
    # 
    #  Part 1: Feedforward the neural network and return the cost in the
    #          variable J. After implementing Part 1, you can verify that your
    #          cost function computation is correct by verifying the cost
    #          computed in ex4.m
    # 
    #  input layer size = number of units in inout layer x1 --- xn
    #  hidden_layer_size = number of units in hidden layers a1 --- an
    #  num_labels = number of outputs a1 --- an is h(x)1 --- h(x)n
    # 
    # 
    #  Expand the 'y' output values into a matrix of single values
    # octave code: y_matrix = eye(num_labels)(y,:);
    y_matrix <- matrix(0,length(y),num_labels)
    for(i in 1:length(y)){y_matrix[i,y[i]] <- 1}
    
    #  a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
    # octave c0de: A1 = [ones(m, 1) X];
    A1 <- cbind(matrix(1,m,1),X)
    #  z2 equals the product of a1 and Theta1 (Theta transpose for product)
    # octave code: Z2 = A1 * Theta1';
    Z2 <- A1 %*% t(Theta1)
    
    #  a2 is the result of passing z2 through g()
    # octave code: A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
    A2 <- cbind(matrix(1,nrow(Z2),1), sigmoid(Z2))
    #  z3 equals product of a2 theta 2 (theta transposed for product)
    # octave code: Z3 = A2*Theta2';
    Z3 <- A2 %*% t(Theta2)
    
    
    #  H(X) = A3 = sigmoid of Z3 and is the ouput of forward propagation
    # octave code: H = A3 = sigmoid(Z3);
    H <- A3 <- sigmoid(Z3)
    
    #  cost Function, non-regularized:
    #  Compute the unregularized cost according to ex4.pdf (top of Page 5), using a3,
    #  your y_matrix, and m (the number of training examples). Note that the 'h' 
    #  argument inside the log() function is exactly a3. Cost should be a scalar value. 
    #  Since y_matrix and a3 are both matrices, you need to compute the double-sum.
    # 
    #  Remember to use element-wise multiplication with the log() function. 
    #  Also, we're using the natural log, not log10().
    # 
    #  Now you can run ex4.m to check the unregularized cost is correct, then you can 
    #  submit this portion to the grader.
    # 
    #  calculate unregularized cost and note that sum( xxxxx, 2) -> 2 is dimension of
    #  summation!
    # octave code: J = (1/m)*sum(sum((-y_matrix).*log(H) - (1-y_matrix).*log(1-H), 2));
    J = (1/m) * sum(sum((-y_matrix)*log(H) - (1-y_matrix)*log(1-H), 2))
    
    #  calculate penalty
    # octave code: penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));
    penalty = (lambda/(2*m))*(sum(colSums(Theta1[, 2:ncol(Theta1)]^2)) + sum(colSums(Theta2[, 2:ncol(Theta2)]^2)))
    
    #  add up cost and regularization
    J = J + penalty;
    
    #  Part 2: Implement the backpropagation algorithm to compute the gradients
    #          Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #          the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #          Theta2_grad, respectively. After implementing Part 2, you can check
    #          that your implementation is correct by running checkNNGradients
    # 
    #          Note: The vector y passed into the function is a vector of labels
    #                containing values from 1..K. You need to map this vector into a 
    #                binary vector of 1's and 0's to be used with the neural network
    #                cost function.
    # 
    #          Hint: We recommend implementing backpropagation using a for-loop
    #                over the training examples if you are implementing it for the 
    #                first time.
    
    #  d3 is the difference between a3 and the y_matrix. The dimensions are the same 
    #  as both, (m x r).
    d3 = A3 - y_matrix;
    
    #  z2 came from the forward propagation process - it's the product of a1 and Theta1, 
    #  prior to applying the sigmoid() function. Dimensions = (m x n) (n x h) --> (m x h)
    # octave code: d2 = (d3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);
    # This can be done in one line of code but to be more clear lets cut it up in multiple
    # lines and peace it together.
    # first compute the matrix multiplication of d3 time theta
    d2 <- d3 %*% Theta2 
    # multiply by sigmoidgradient of Z2 with added ones in column 1 (cbind) = bias
    d2 <- d2 * sigmoidGradient(cbind(matrix(1, nrow(Z2), 1), Z2))
    # delete the first column (bias) to get final d2
    d2 <- d2[,2:ncol(d2)]
    
    # compute D1 by transpose d2 matrix multiply by A1
    # octave code: D1 = d2'*A1;
    D1 = t(d2) %*% A1
    D2 = t(d3) %*% A2
    
    
    #  Part 3: Implement regularization with the cost function and gradients.
    # 
    #          Hint: You can implement this around the code for
    #                backpropagation. That is, you can compute the gradients for
    #                the regularization separately and then add them to Theta1_grad
    #                and Theta2_grad from Part 2.
    # 
    # octave code: Theta1_grad = D1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
    Theta1_grad = D1/m + ((lambda/m) * cbind(matrix(0,nrow(Theta1),1), Theta1[, 2:ncol(Theta1)]))
    # octave code: Theta2_grad = D2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
    Theta2_grad = D2/m + ((lambda/m) * cbind(matrix(0,nrow(Theta2),1), Theta2[, 2:ncol(Theta2)]))
    
    #  =========================================================================
    # 
    #  Unroll gradients
    # ocatve coce: grad = [Theta1_grad(:) ; Theta2_grad(:)];
    grad <- c(c(Theta1_grad), c(Theta2_grad))
    return(list(J,grad))
}

################################################################################
# Main part                                                                    #
################################################################################

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
# (note that we have mapped "0" to label 10)


# =========== Part 1: Loading and Visualizing Data =============
#   We start the exercise by first loading and visualizing the dataset. 
#   You will be working with a dataset that contains handwritten digits.

print("Loading and Visualizing Data")
# octave code: load('ex4data1.mat') which outputs X and y matrices
ls_data <- readMat('ex4data1.mat', fixNames=TRUE) # returns list of matrices X and y
X <- ls_data$X; y <- ls_data$y; rm(ls_data)

# get nrow octave code: m = size(X, 1);
m <- nrow(X)

# Randomly select 100 data points to display
# octave code: sel = randperm(size(X, 1));
# octave code: sel = sel(1:100);
sel <- X[sample(nrow(X),size=100,replace=FALSE),]

# display images 10 x 10 grid
# octave code: displayData(X(sel, :));
image(display_data(sel, 20,20))


# ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.
print("Loading Saved Neural Network Parameters")
# Load the weights into variables Theta1 and Theta2
# octave code: load('ex4weights.mat');
ls_data <- readMat('ex4weights.mat', fixNames=TRUE) # returns list with theta1 and theta2
Theta1 <- ls_data$Theta1; Theta2 <- ls_data$Theta2
rm(ls_data) # clean up unnecessary list

# Unroll parameters (vectorize matrices and concatenate them)
# octave code: nn_params = [Theta1(:) ; Theta2(:)];
nn_params <- c(c(Theta1),c(Theta2))


#  ================ Part 3: Compute Cost (Feedforward) ================
#       To the neural network, you should first start by implementing the
#   feedforward part of the neural network that returns the cost only. You
#   should complete the code in nnCostFunction.m to return cost. After
#   implementing the feedforward to compute the cost, you can verify that
#   your implementation is correct by verifying that you get the same cost
#   as us for the fixed debugging parameters.
# 
#   We suggest implementing the feedforward cost *without* regularization
#   first so that it will be easier for you to debug. Later, in part 4, you
#   will get to implement the regularized cost.

print("Feedforward Using Neural Network ")
# Weight regularization parameter (we set this to 0 here).
# octave code: lambda = 0;
lambda <- 0

# compute the cost
# ocatve code: J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
#                   num_labels, X, y, lambda);
J <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)[[1]]

print("Cost at parameters (loaded from ex4weights) value should be about 0.287629")
print(J)

# =============== Part 4: Implement Regularization ===============
# Once your cost function implementation is correct, you should now
# continue to implement the regularization with the cost.

# Weight regularization parameter (we set this to 1 here).
# octave code: lambda = 1;
lambda <- 1

# octave code: J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
#                   num_labels, X, y, lambda);
J <- nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)[[1]]

# octave code: fprintf(['Cost at parameters (loaded from ex4weights): #f '...
#         '\n(this value should be about 0.383770)\n'], J);
print("Cost at parameters (loaded from ex4weights, should about 0.383770 in R 0.3841699")
print(J)


# ================ Part 5: Sigmoid Gradient  ================
# Before you start implementing the neural network, you will first
# implement the gradient for the sigmoid function. You should complete the
# code in the sigmoidGradient.m file.

print("Evaluating sigmoid gradient")

# octave code: g = sigmoidGradient([1 -0.5 0 0.5 1]);
g <- sigmoidGradient(c(1,-0.5, 0, 0.5, 1))
print("Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]")
print(g)
print("should be about: 0.196612 0.235004 0.250000 0.235004 0.196612")


# ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print("Initializing Neural Network Parameters ...")

# octave code: initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
# octave code: initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
# octave code: initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
initial_nn_params <- c(c(initial_Theta1), c(initial_Theta2))


## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print("Checking Backpropagation... ")

#  Check gradients by running checkNNGradients
# octave code: checkNNGradients; not initializing parameter lambda means lambda = 0
checkNNGradients(lambda = 0) # explicitly stating lambda = 0 is more clear!


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('Training Neural Network...')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.
options <- optimset('MaxIter', 50); <-----------------------------------------------------

#  You should also try different values of lambda
lambda = 1;

# Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


####################################################################################################
library(optimx)

costFun <- nnCost(nn_params = initial_nn_params,
                  input_layer_size = input_layer_size,
                  hidden_layer_size = hidden_layer_size,
                  num_labels = num_labels,
                  X = X, y = y, lambda = lambda)

optim(initial_nn_params, fn=nnCost_short, method = c("L-BFGS-B"), control = c("maxit = 50"))

for(i in 1:100){
    par_s <- initial_nn_params - i* 0.01
    print(nnCost_short(par_s))
}



mydat <- rnorm(100, 5, 20)

ll <- function(pars, dat){
    -sum(dnorm(dat, pars[1], pars[2], log = TRUE))
}

optimx(c(5, 20), ll, dat = mydat)
# The 'true' MLEs
mean(mydat)
sqrt(var(mydat)*(99/100))


####################################################################################################
# Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: #f\n', mean(double(pred == y)) * 100);










#################################################################################################
################################################################################
# nnCost function                                                              #
# use = nnCost(nn_params, input_layer_size, hidden_layer_size,                 #
# num_labels, X, y, lambda)                                                    #
# output = a list with the cost (one value) for Theta1 and Theta2              #
# test: J <- nnCost(nn_params, input_layer_size, hidden_layer_size,            #
# num_labels, X, y, lambda)                                                    #
################################################################################
nnCost <- function(nn_params, input_layer_size, hidden_layer_size, 
                           num_labels, X, y, lambda){
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #    X, y, lambda) computes the cost and gradient of the neural network. The
    #    parameters for the neural network are "unrolled" into the vector
    #    nn_params and need to be converted back into the weight matrices. 
    #  
    #    The returned parameter grad should be a "unrolled" vector of the
    #    partial derivatives of the neural network.
    # 
    # 
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    # octave code: Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    #                  hidden_layer_size, (input_layer_size + 1));
    Theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
                     hidden_layer_size,(input_layer_size + 1))
    # octave code: Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    #                  num_labels, (hidden_layer_size + 1));
    Theta2 <- matrix(nn_params[(1+(hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
                     num_labels,(hidden_layer_size + 1))
    
    #  Setup some useful variables
    m <- nrow(X)
    # 
    #  You need to return the following variables correctly 
    J <- 0
    # octave code: Theta1_grad = zeros(size(Theta1));
    Theta1_grad <- matrix(0,nrow(Theta1), ncol(Theta1))
    # octave code: Theta2_grad = zeros(size(Theta2));
    Theta2_grad <- matrix(0,nrow(Theta2), ncol(Theta2))
    
    #  ====================== YOUR CODE HERE ======================
    #      Instructions: You should complete the code by working through the
    #                following parts.
    # 
    #  Part 1: Feedforward the neural network and return the cost in the
    #          variable J. After implementing Part 1, you can verify that your
    #          cost function computation is correct by verifying the cost
    #          computed in ex4.m
    # 
    #  input layer size = number of units in inout layer x1 --- xn
    #  hidden_layer_size = number of units in hidden layers a1 --- an
    #  num_labels = number of outputs a1 --- an is h(x)1 --- h(x)n
    # 
    # 
    #  Expand the 'y' output values into a matrix of single values
    # octave code: y_matrix = eye(num_labels)(y,:);
    y_matrix <- matrix(0,length(y),num_labels)
    for(i in 1:length(y)){y_matrix[i,y[i]] <- 1}
    
    #  a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
    # octave c0de: A1 = [ones(m, 1) X];
    A1 <- cbind(matrix(1,m,1),X)
    #  z2 equals the product of a1 and Theta1 (Theta transpose for product)
    # octave code: Z2 = A1 * Theta1';
    Z2 <- A1 %*% t(Theta1)
    
    #  a2 is the result of passing z2 through g()
    # octave code: A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
    A2 <- cbind(matrix(1,nrow(Z2),1), sigmoid(Z2))
    #  z3 equals product of a2 theta 2 (theta transposed for product)
    # octave code: Z3 = A2*Theta2';
    Z3 <- A2 %*% t(Theta2)
    
    
    #  H(X) = A3 = sigmoid of Z3 and is the ouput of forward propagation
    # octave code: H = A3 = sigmoid(Z3);
    H <- A3 <- sigmoid(Z3)
    
    #  cost Function, non-regularized:
    #  Compute the unregularized cost according to ex4.pdf (top of Page 5), using a3,
    #  your y_matrix, and m (the number of training examples). Note that the 'h' 
    #  argument inside the log() function is exactly a3. Cost should be a scalar value. 
    #  Since y_matrix and a3 are both matrices, you need to compute the double-sum.
    # 
    #  Remember to use element-wise multiplication with the log() function. 
    #  Also, we're using the natural log, not log10().
    # 
    #  Now you can run ex4.m to check the unregularized cost is correct, then you can 
    #  submit this portion to the grader.
    # 
    #  calculate unregularized cost and note that sum( xxxxx, 2) -> 2 is dimension of
    #  summation!
    # octave code: J = (1/m)*sum(sum((-y_matrix).*log(H) - (1-y_matrix).*log(1-H), 2));
    J = (1/m) * sum(sum((-y_matrix)*log(H) - (1-y_matrix)*log(1-H), 2))
    
    #  calculate penalty
    # octave code: penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));
    penalty = (lambda/(2*m))*(sum(colSums(Theta1[, 2:ncol(Theta1)]^2)) + sum(colSums(Theta2[, 2:ncol(Theta2)]^2)))
    
    #  add up cost and regularization
    J = J + penalty;

    return(J)
}

################################################################################
# nnGrad function                                                              #
# use = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,         #
# num_labels, X, y, lambda)                                                    #
# output = an unrolled vector of gradients for Theta1 and Theta2               #
# test: J <- nnGrad(nn_params, input_layer_size, hidden_layer_size,            #
# num_labels, X, y, lambda)                                                    #
################################################################################
nnGrad <- function(nn_params, input_layer_size, hidden_layer_size, 
                           num_labels, X, y, lambda){
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #    X, y, lambda) computes the cost and gradient of the neural network. The
    #    parameters for the neural network are "unrolled" into the vector
    #    nn_params and need to be converted back into the weight matrices. 
    #  
    #    The returned parameter grad should be a "unrolled" vector of the
    #    partial derivatives of the neural network.
    # 
    # 
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    # octave code: Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    #                  hidden_layer_size, (input_layer_size + 1));
    Theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
                     hidden_layer_size,(input_layer_size + 1))
    # octave code: Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    #                  num_labels, (hidden_layer_size + 1));
    Theta2 <- matrix(nn_params[(1+(hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
                     num_labels,(hidden_layer_size + 1))
    
    #  Setup some useful variables
    m <- nrow(X)
    # 
    #  You need to return the following variables correctly 
    J <- 0
    # octave code: Theta1_grad = zeros(size(Theta1));
    Theta1_grad <- matrix(0,nrow(Theta1), ncol(Theta1))
    # octave code: Theta2_grad = zeros(size(Theta2));
    Theta2_grad <- matrix(0,nrow(Theta2), ncol(Theta2))
    
    #  ====================== YOUR CODE HERE ======================
    #      Instructions: You should complete the code by working through the
    #                following parts.
    # 
    #  Part 1: Feedforward the neural network and return the cost in the
    #          variable J. After implementing Part 1, you can verify that your
    #          cost function computation is correct by verifying the cost
    #          computed in ex4.m
    # 
    #  input layer size = number of units in inout layer x1 --- xn
    #  hidden_layer_size = number of units in hidden layers a1 --- an
    #  num_labels = number of outputs a1 --- an is h(x)1 --- h(x)n
    # 
    # 
    #  Expand the 'y' output values into a matrix of single values
    # octave code: y_matrix = eye(num_labels)(y,:);
    y_matrix <- matrix(0,length(y),num_labels)
    for(i in 1:length(y)){y_matrix[i,y[i]] <- 1}
    
    #  a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
    # octave c0de: A1 = [ones(m, 1) X];
    A1 <- cbind(matrix(1,m,1),X)
    #  z2 equals the product of a1 and Theta1 (Theta transpose for product)
    # octave code: Z2 = A1 * Theta1';
    Z2 <- A1 %*% t(Theta1)
    
    #  a2 is the result of passing z2 through g()
    # octave code: A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
    A2 <- cbind(matrix(1,nrow(Z2),1), sigmoid(Z2))
    #  z3 equals product of a2 theta 2 (theta transposed for product)
    # octave code: Z3 = A2*Theta2';
    Z3 <- A2 %*% t(Theta2)
    
    
    #  H(X) = A3 = sigmoid of Z3 and is the ouput of forward propagation
    # octave code: H = A3 = sigmoid(Z3);
    H <- A3 <- sigmoid(Z3)
    
    #  Part 2: Implement the backpropagation algorithm to compute the gradients
    #          Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #          the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #          Theta2_grad, respectively. After implementing Part 2, you can check
    #          that your implementation is correct by running checkNNGradients
    # 
    #          Note: The vector y passed into the function is a vector of labels
    #                containing values from 1..K. You need to map this vector into a 
    #                binary vector of 1's and 0's to be used with the neural network
    #                cost function.
    # 
    #          Hint: We recommend implementing backpropagation using a for-loop
    #                over the training examples if you are implementing it for the 
    #                first time.
    
    #  d3 is the difference between a3 and the y_matrix. The dimensions are the same 
    #  as both, (m x r).
    d3 = A3 - y_matrix;
    
    #  z2 came from the forward propagation process - it's the product of a1 and Theta1, 
    #  prior to applying the sigmoid() function. Dimensions = (m x n) (n x h) --> (m x h)
    # octave code: d2 = (d3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);
    # This can be done in one line of code but to be more clear lets cut it up in multiple
    # lines and peace it together.
    # first compute the matrix multiplication of d3 time theta
    d2 <- d3 %*% Theta2 
    # multiply by sigmoidgradient of Z2 with added ones in column 1 (cbind) = bias
    d2 <- d2 * sigmoidGradient(cbind(matrix(1, nrow(Z2), 1), Z2))
    # delete the first column (bias) to get final d2
    d2 <- d2[,2:ncol(d2)]
    
    # compute D1 by transpose d2 matrix multiply by A1
    # octave code: D1 = d2'*A1;
    D1 = t(d2) %*% A1
    D2 = t(d3) %*% A2
    
    
    #  Part 3: Implement regularization with the cost function and gradients.
    # 
    #          Hint: You can implement this around the code for
    #                backpropagation. That is, you can compute the gradients for
    #                the regularization separately and then add them to Theta1_grad
    #                and Theta2_grad from Part 2.
    # 
    # octave code: Theta1_grad = D1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
    Theta1_grad = D1/m + ((lambda/m) * cbind(matrix(0,nrow(Theta1),1), Theta1[, 2:ncol(Theta1)]))
    # octave code: Theta2_grad = D2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
    Theta2_grad = D2/m + ((lambda/m) * cbind(matrix(0,nrow(Theta2),1), Theta2[, 2:ncol(Theta2)]))
    
    #  =========================================================================
    # 
    #  Unroll gradients
    # ocatve coce: grad = [Theta1_grad(:) ; Theta2_grad(:)];
    grad <- c(c(Theta1_grad), c(Theta2_grad))
    return(grad)
}
################################################################################
nnCost_short <- function(nn_params){
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #    X, y, lambda) computes the cost and gradient of the neural network. The
    #    parameters for the neural network are "unrolled" into the vector
    #    nn_params and need to be converted back into the weight matrices. 
    #  
    #    The returned parameter grad should be a "unrolled" vector of the
    #    partial derivatives of the neural network.
    # 
    # 
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    # octave code: Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    #                  hidden_layer_size, (input_layer_size + 1));
    Theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
                     hidden_layer_size,(input_layer_size + 1))
    # octave code: Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    #                  num_labels, (hidden_layer_size + 1));
    Theta2 <- matrix(nn_params[(1+(hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
                     num_labels,(hidden_layer_size + 1))
    
    #  Setup some useful variables
    m <- nrow(X)
    # 
    #  You need to return the following variables correctly 
    J <- 0
    # octave code: Theta1_grad = zeros(size(Theta1));
    Theta1_grad <- matrix(0,nrow(Theta1), ncol(Theta1))
    # octave code: Theta2_grad = zeros(size(Theta2));
    Theta2_grad <- matrix(0,nrow(Theta2), ncol(Theta2))
    
    #  ====================== YOUR CODE HERE ======================
    #      Instructions: You should complete the code by working through the
    #                following parts.
    # 
    #  Part 1: Feedforward the neural network and return the cost in the
    #          variable J. After implementing Part 1, you can verify that your
    #          cost function computation is correct by verifying the cost
    #          computed in ex4.m
    # 
    #  input layer size = number of units in inout layer x1 --- xn
    #  hidden_layer_size = number of units in hidden layers a1 --- an
    #  num_labels = number of outputs a1 --- an is h(x)1 --- h(x)n
    # 
    # 
    #  Expand the 'y' output values into a matrix of single values
    # octave code: y_matrix = eye(num_labels)(y,:);
    y_matrix <- matrix(0,length(y),num_labels)
    for(i in 1:length(y)){y_matrix[i,y[i]] <- 1}
    
    #  a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
    # octave c0de: A1 = [ones(m, 1) X];
    A1 <- cbind(matrix(1,m,1),X)
    #  z2 equals the product of a1 and Theta1 (Theta transpose for product)
    # octave code: Z2 = A1 * Theta1';
    Z2 <- A1 %*% t(Theta1)
    
    #  a2 is the result of passing z2 through g()
    # octave code: A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
    A2 <- cbind(matrix(1,nrow(Z2),1), sigmoid(Z2))
    #  z3 equals product of a2 theta 2 (theta transposed for product)
    # octave code: Z3 = A2*Theta2';
    Z3 <- A2 %*% t(Theta2)
    
    
    #  H(X) = A3 = sigmoid of Z3 and is the ouput of forward propagation
    # octave code: H = A3 = sigmoid(Z3);
    H <- A3 <- sigmoid(Z3)
    
    #  cost Function, non-regularized:
    #  Compute the unregularized cost according to ex4.pdf (top of Page 5), using a3,
    #  your y_matrix, and m (the number of training examples). Note that the 'h' 
    #  argument inside the log() function is exactly a3. Cost should be a scalar value. 
    #  Since y_matrix and a3 are both matrices, you need to compute the double-sum.
    # 
    #  Remember to use element-wise multiplication with the log() function. 
    #  Also, we're using the natural log, not log10().
    # 
    #  Now you can run ex4.m to check the unregularized cost is correct, then you can 
    #  submit this portion to the grader.
    # 
    #  calculate unregularized cost and note that sum( xxxxx, 2) -> 2 is dimension of
    #  summation!
    # octave code: J = (1/m)*sum(sum((-y_matrix).*log(H) - (1-y_matrix).*log(1-H), 2));
    J = (1/m) * sum(sum((-y_matrix)*log(H) - (1-y_matrix)*log(1-H), 2))
    
    #  calculate penalty
    # octave code: penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));
    penalty = (lambda/(2*m))*(sum(colSums(Theta1[, 2:ncol(Theta1)]^2)) + sum(colSums(Theta2[, 2:ncol(Theta2)]^2)))
    
    #  add up cost and regularization
    J = J + penalty;
    
    return(J)
}
################################################################################
nnGrad_short <- function(nn_params){
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #    X, y, lambda) computes the cost and gradient of the neural network. The
    #    parameters for the neural network are "unrolled" into the vector
    #    nn_params and need to be converted back into the weight matrices. 
    #  
    #    The returned parameter grad should be a "unrolled" vector of the
    #    partial derivatives of the neural network.
    # 
    # 
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    # octave code: Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    #                  hidden_layer_size, (input_layer_size + 1));
    Theta1 <- matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))],
                     hidden_layer_size,(input_layer_size + 1))
    # octave code: Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    #                  num_labels, (hidden_layer_size + 1));
    Theta2 <- matrix(nn_params[(1+(hidden_layer_size * (input_layer_size + 1))):length(nn_params)],
                     num_labels,(hidden_layer_size + 1))
    
    #  Setup some useful variables
    m <- nrow(X)
    # 
    #  You need to return the following variables correctly 
    J <- 0
    # octave code: Theta1_grad = zeros(size(Theta1));
    Theta1_grad <- matrix(0,nrow(Theta1), ncol(Theta1))
    # octave code: Theta2_grad = zeros(size(Theta2));
    Theta2_grad <- matrix(0,nrow(Theta2), ncol(Theta2))
    
    #  ====================== YOUR CODE HERE ======================
    #      Instructions: You should complete the code by working through the
    #                following parts.
    # 
    #  Part 1: Feedforward the neural network and return the cost in the
    #          variable J. After implementing Part 1, you can verify that your
    #          cost function computation is correct by verifying the cost
    #          computed in ex4.m
    # 
    #  input layer size = number of units in inout layer x1 --- xn
    #  hidden_layer_size = number of units in hidden layers a1 --- an
    #  num_labels = number of outputs a1 --- an is h(x)1 --- h(x)n
    # 
    # 
    #  Expand the 'y' output values into a matrix of single values
    # octave code: y_matrix = eye(num_labels)(y,:);
    y_matrix <- matrix(0,length(y),num_labels)
    for(i in 1:length(y)){y_matrix[i,y[i]] <- 1}
    
    #  a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
    # octave c0de: A1 = [ones(m, 1) X];
    A1 <- cbind(matrix(1,m,1),X)
    #  z2 equals the product of a1 and Theta1 (Theta transpose for product)
    # octave code: Z2 = A1 * Theta1';
    Z2 <- A1 %*% t(Theta1)
    
    #  a2 is the result of passing z2 through g()
    # octave code: A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
    A2 <- cbind(matrix(1,nrow(Z2),1), sigmoid(Z2))
    #  z3 equals product of a2 theta 2 (theta transposed for product)
    # octave code: Z3 = A2*Theta2';
    Z3 <- A2 %*% t(Theta2)
    
    
    #  H(X) = A3 = sigmoid of Z3 and is the ouput of forward propagation
    # octave code: H = A3 = sigmoid(Z3);
    H <- A3 <- sigmoid(Z3)
    
    #  Part 2: Implement the backpropagation algorithm to compute the gradients
    #          Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #          the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #          Theta2_grad, respectively. After implementing Part 2, you can check
    #          that your implementation is correct by running checkNNGradients
    # 
    #          Note: The vector y passed into the function is a vector of labels
    #                containing values from 1..K. You need to map this vector into a 
    #                binary vector of 1's and 0's to be used with the neural network
    #                cost function.
    # 
    #          Hint: We recommend implementing backpropagation using a for-loop
    #                over the training examples if you are implementing it for the 
    #                first time.
    
    #  d3 is the difference between a3 and the y_matrix. The dimensions are the same 
    #  as both, (m x r).
    d3 = A3 - y_matrix;
    
    #  z2 came from the forward propagation process - it's the product of a1 and Theta1, 
    #  prior to applying the sigmoid() function. Dimensions = (m x n) (n x h) --> (m x h)
    # octave code: d2 = (d3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);
    # This can be done in one line of code but to be more clear lets cut it up in multiple
    # lines and peace it together.
    # first compute the matrix multiplication of d3 time theta
    d2 <- d3 %*% Theta2 
    # multiply by sigmoidgradient of Z2 with added ones in column 1 (cbind) = bias
    d2 <- d2 * sigmoidGradient(cbind(matrix(1, nrow(Z2), 1), Z2))
    # delete the first column (bias) to get final d2
    d2 <- d2[,2:ncol(d2)]
    
    # compute D1 by transpose d2 matrix multiply by A1
    # octave code: D1 = d2'*A1;
    D1 = t(d2) %*% A1
    D2 = t(d3) %*% A2
    
    
    #  Part 3: Implement regularization with the cost function and gradients.
    # 
    #          Hint: You can implement this around the code for
    #                backpropagation. That is, you can compute the gradients for
    #                the regularization separately and then add them to Theta1_grad
    #                and Theta2_grad from Part 2.
    # 
    # octave code: Theta1_grad = D1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
    Theta1_grad = D1/m + ((lambda/m) * cbind(matrix(0,nrow(Theta1),1), Theta1[, 2:ncol(Theta1)]))
    # octave code: Theta2_grad = D2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
    Theta2_grad = D2/m + ((lambda/m) * cbind(matrix(0,nrow(Theta2),1), Theta2[, 2:ncol(Theta2)]))
    
    #  =========================================================================
    # 
    #  Unroll gradients
    # ocatve coce: grad = [Theta1_grad(:) ; Theta2_grad(:)];
    grad <- c(c(Theta1_grad), c(Theta2_grad))
    return(grad)
}




# test stuff
library(ggplot2)
plot_image <- function(row) {
    photo <- data.frame( x=rep(1:20,times=20), y=rep(20:1,each=20), shade=as.numeric(sel[row,]))
    ggplot(data=photo) + geom_point(aes(x=x,y=y,color=shade), size=11, shape=15) + 
        theme( axis.line=element_blank(), axis.text.x=element_blank(), 
               axis.text.y=element_blank(), axis.ticks=element_blank(), 
               axis.title.x=element_blank(), axis.title.y=element_blank(), 
               legend.position="none", panel.background=element_blank(), 
               panel.border=element_blank(), panel.grid.major=element_blank(), 
               panel.grid.minor=element_blank(), plot.background=element_blank()) + 
        scale_color_gradient(low="white",high="black")
}
par(ask=TRUE)
lapply(sample(nrow(sel),size=50), FUN=plot_image)
par(ask=FALSE)