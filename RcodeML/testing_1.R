################################################################################
# ML copy of class method

# initialize octave code: clear ; close all; clc
rm(list = ls())

# load library to read matlab files
library(R.matlab)

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
        
        #print(paste("image:", i,"row:", img_all_row,"col:", img_all_col, sep = " "))
        # place image in the final matrix
        img_all[img_all_row:(img_all_row + row_nr-1),img_all_col:(img_all_col + col_nr-1)] <- img[1:20,1:20]
    }
    
    # R plot rotates 90 degrees CW so function rotates back
    return(t(apply(img_all, 2, rev)))
}
# display the data
image(display_data(sel, 20,20))






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
# sigmoid function                                                             #
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










# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
# (note that we have mapped "0" to label 10)










#################################################################################################
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