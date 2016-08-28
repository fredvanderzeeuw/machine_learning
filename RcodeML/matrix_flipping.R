V <- c(1,2,3,4,5,6)
X <- matrix(V, nrow=3, ncol=2)
Y <- matrix(V, nrow=3, ncol=2)

## add matrices
X + Y

## cbind matrices
cbind(X,Y)
## rbind matrices
rbind(X,Y)

## flip x and flip y
X <- matrix(data=c(1,2,3,4,5,6), nrow = 3, ncol = 2)
matrix(rev(X), nrow = nrow(X), ncol = ncol(X))

## transpose = 90 degrees clockwise
X
t(rev(X))

## flip x
X_flip_x <- matrix(nrow = nrow(X), ncol = ncol(X))
X_flip_x[,1:ncol(X)] <- X[,ncol(X):1]
X_flip_x

## flip y
X_flip_y <- matrix(nrow = nrow(X), ncol = ncol(X))
X_flip_y[1:nrow(X),] <- X[nrow(X):1,]
X_flip_y

## 180 degrees clockwise
X
matrix(rev(X), nrow = nrow(X), ncol = ncol(X))