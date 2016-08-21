clear; close all; clc;

% test code for function
 X = [ -1 -1 ; -1 -2 ; -2 -1 ; -2 -2 ; ...
          1 1 ;  1 2 ;  2 1 ; 2 2 ; ...
         -1 1 ;  -1 2 ;  -2 1 ; -2 2 ; ...
          1 -1 ; 1 -2 ;  -2 -1 ; -2 -2 ];
y = [ 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 ]';

  
all_theta = oneVsAll(X, y, 4, 0.1)
predictOneVsAll(all_theta, X)



clear; close all; clc
Xm = [ -1 -1 ; -1 -2 ; -2 -1 ; -2 -2 ; ...
          1 1 ;  1 2 ;  2 1 ; 2 2 ; ...
         -1 1 ;  -1 2 ;  -2 1 ; -2 2 ; ...
          1 -1 ; 1 -2 ;  -2 -1 ; -2 -2 ];
ym = [ 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 ]';
t1 = sin(reshape(1:2:24, 4, 3));
t2 = cos(reshape(1:2:40, 4, 5));

Theta1 = t1; Theta2 = t2; X = Xm
predict(t1, t2, Xm)