>> A = [1 2;3 4;5 6]
A =

   1   2
   3   4
   5   6

>> size(A)
ans =

   3   2

>> length(A)
ans =  3
>> clear
>> v = [1 2 3 4] % create vector 1 * 4
v =

   1   2   3   4

>> size(v) % get size of A
ans =

   1   4

>> % meaning 1 row and 4 columns
>> v = [1;2;3;4] % create vector 4 * 1
v =

   1
   2
   3
   4
   
>> size(v)
ans =

   4   1

>> % create v stepwise 1 - 2 in steps 0.1:
>> v = 1:0.1:2
v =

    1.0000    1.1000    1.2000    1.3000    1.4000    1.5000    1.6000    1.7000    1.8000    1.9000    2.0000

>> % create a vector with ones:
>> ones(2,3)
ans =

   1   1   1
   1   1   1

>> % or two's:
>> 2*ones(2,3)
ans =

   2   2   2
   2   2   2

>> zeros(2,3)
ans =

   0   0   0
   0   0   0

>> eye(2,3)
ans =

Diagonal Matrix

   1   0   0
   0   1   0

>> eye(3,3)
ans =

Diagonal Matrix

   1   0   0
   0   1   0
   0   0   1
>> w = randn(2,3) % random values
w =

   2.30015  -1.17471   0.93808
  -0.64149  -0.22331  -2.31431

>> % create vector and histogram:
>> w = -6 + sqrt(10) * randn(1,1000);
>> hist(w)
>> hist(w,20) % 20 bins
>> hist(w,40) % 40 bins
>> help eye % get help

>> A = [1 2; 3 4; 5 6]
A =

   1   2
   3   4
   5   6

>> size(A)
ans =

   3   2

>> sz = size(A)
sz =

   3   2

>> size(sz)
ans =

   1   2

>> % last size gives size of var sz which is a 1 by 2 matrix
>> size(A,1) % size of rows
ans =  3
>> size(A,2) % size of columns
ans =  2
>> v
v =

    1.0000    1.1000    1.2000    1.3000    1.4000    1.5000    1.6000    1.7000    1.8000    1.9000    2.0000

>> length(v)
ans =  11
>> % meaning 11 elements
>> length(A) % longest length of A is number of rows
ans =  3
>> % get current path
>> pwd
ans = C:\Users\fredv\Documents\code\ML_Stanford
>> % cd 'C:\Users\fredv\Documents\code\ML_Stanford' % change to dir

>> ls % give file list
 Volume in drive C is OS
 Volume Serial Number is A085-6A03

 Directory of C:\Users\fredv\Documents\code\ML_Stanford

[.]                    [homeworkw2]           try_1.m
[..]                   [machine-learning-ex1]
               1 File(s)             50 bytes
               4 Dir(s)  81.639.215.104 bytes free
>> who
Variables in the current scope:

A    ans  sz   v    w

>> whos
Variables in the current scope:

   Attr Name        Size                     Bytes  Class
   ==== ====        ====                     =====  =====
        A           3x2                         48  double
        ans         1x41                        41  char
        sz          1x2                         16  double
        v           1x11                        24  double
        w           1x1000                    8000  double

Total is 1060 elements using 8129 bytes

>> % clear A will delete var A
>> clear with no var attached will delete all vars
>> % clear with no var attached will delete all vars

>> [a, b] = textread ('Map1.txt', "%f %f")
% loads file 

% load file columns in vectors
[a, b] = textread ('Map1.txt', "%f %f")

% read file with ignoring header line
 X = dlmread (filename,',',1,0)  
 
 >> Y = X(:,[2 3])
 >> size(Y)
ans =

   20    2

>> size(X)
ans =

   20    3


>> A=[1 2; 3 4; 5 6]
A =

   1   2
   3   4
   5   6

>> B=[11 12; 13 14; 15 16]
B =

   11   12
   13   14
   15   16

>> C=[A B]
C =

    1    2   11   12
    3    4   13   14
    5    6   15   16

>> C=[A;B]
C =

    1    2
    3    4
    5    6
   11   12
   13   14
   15   16

>> % [A B] is wide append while [A;B] is rowbind

% matrix calc by elements
A = [1, 6, 3; 2, 7, 4]
B = [2, 7, 2; 7, 3, 9]
A ./ B
A .* B

% multiply a matrix in the normal way (not element by element
X = [1 3; 4 0; 2 1]
v = [1;5]
X * v

A=[1 2; 3 4; 5 6]
B=[11 12; 13 14; 15 16]
C=[1 1; 2 2]

>> V=[1;2;3]
V =

   1
   2
   3

>> 1 ./V
ans =

   1.00000
   0.50000
   0.33333

>> 1 .? A
error: invalid character '?' (ASCII 63) near line 1, column 5
parse error:

  syntax error

>>> 1 .? A
       ^

>> 1 ./ A
ans =

   1.00000   0.50000
   0.33333   0.25000
   0.20000   0.16667

>> log(V)
ans =

   0.00000
   0.69315
   1.09861

>> exp(V)
ans =

    2.7183
    7.3891
   20.0855

>> abs(V)
ans =

   1
   2
   3
   
>> -V
ans =

  -1
  -2
  -3

>> V + ones(length(V),1)
ans =

   2
   3
   4

>> length(v)
ans =  2
>> ones(3,1)
ans =

   1
   1
   1

>> V + 1
ans =

   2
   3
   4

>> % same result
>> A
A =

   1   2
   3   4
   5   6
   
>> A'
ans =

   1   3   5
   2   4   6

>> a=[1 15 2 0.5]
a =

    1.00000   15.00000    2.00000    0.50000

>> val= max(a)
val =  15
>> [val, ind], max(a)
error: 'ind' undefined near line 1 column 6
>> [val, ind]= max(a)
val =  15
ind =  2
>> % ind here stands for index of value that was max
>> max(A) % will give max of a column!!!
ans =

   5   6

>> a <  3
ans =

   1   0   1   1

>> sum(a <  3)
ans =  3
>> find(a < 3)
ans =

   1   3   4
   
>> A = magic(3)
A =

   8   1   6
   3   5   7
   4   9   2

>> [r,c] = find(A >= 7)
r =

   1
   3
   2

c =

   1
   2
   3

>> sum(a)
ans =  18.500
>> prod(a)
ans =  15
>> floor(a)
ans =

    1   15    2    0

>> ceiling(a)
error: 'ceiling' undefined near line 1 column 1
>> ceil(a)
ans =

    1   15    2    1

>> A = magic(3)
A =

   8   1   6
   3   5   7
   4   9   2

>> [r,c] = find(A >= 7)
r =

   1
   3
   2

c =

   1
   2
   3

>> sum(a)
ans =  18.500
>> prod(a)
ans =  15
>> floor(a)
ans =

    1   15    2    0

>> ceiling(a)
error: 'ceiling' undefined near line 1 column 1
>> ceil(a)
ans =

    1   15    2    1
>> A=magic(9)
A =

   47   58   69   80    1   12   23   34   45
   57   68   79    9   11   22   33   44   46
   67   78    8   10   21   32   43   54   56
   77    7   18   20   31   42   53   55   66
    6   17   19   30   41   52   63   65   76
   16   27   29   40   51   62   64   75    5
   26   28   39   50   61   72   74    4   15
   36   38   49   60   71   73    3   14   25
   37   48   59   70   81    2   13   24   35

>> sum(A,1)
ans =

   369   369   369   369   369   369   369   369   369

>> sum(A,2)
ans =

   369
   369
   369
   369
   369
   369
   369
   369
   369

>> eye(9)
ans =

Diagonal Matrix

   1   0   0   0   0   0   0   0   0
   0   1   0   0   0   0   0   0   0
   0   0   1   0   0   0   0   0   0
   0   0   0   1   0   0   0   0   0
   0   0   0   0   1   0   0   0   0
   0   0   0   0   0   1   0   0   0
   0   0   0   0   0   0   1   0   0
   0   0   0   0   0   0   0   1   0
   0   0   0   0   0   0   0   0   1

>> A .* eye(9)
ans =

   47    0    0    0    0    0    0    0    0
    0   68    0    0    0    0    0    0    0
    0    0    8    0    0    0    0    0    0
    0    0    0   20    0    0    0    0    0
    0    0    0    0   41    0    0    0    0
    0    0    0    0    0   62    0    0    0
    0    0    0    0    0    0   74    0    0
    0    0    0    0    0    0    0   14    0
    0    0    0    0    0    0    0    0   35

>> sum(sum(A .* eye(9)))
ans =  369
>> A=magic(3)
A =

   8   1   6
   3   5   7
   4   9   2

>> pinv(A)
ans =

   0.147222  -0.144444   0.063889
  -0.061111   0.022222   0.105556
  -0.019444   0.188889  -0.102778

>> temp = pinv(A)
temp =

   0.147222  -0.144444   0.063889
  -0.061111   0.022222   0.105556
  -0.019444   0.188889  -0.102778

>> temp*A
ans =

   1.00000   0.00000  -0.00000
  -0.00000   1.00000   0.00000
   0.00000   0.00000   1.00000

 

 >> % plotting data
>> t=[0:0.1:0.98]
t =

   0.00000   0.10000   0.20000   0.30000   0.40000   0.50000   0.60000   0.70000   0.80000   0.90000

>> t=[0:0.01:0.98];
>> y1=sin(2*pi*4*t);
>> plot(t,y1)
>> % t = x-axis y = y-axis
>> y2=cos(2*pi*4*t);
>> plot(t,y2)
>> % now plot y1 and y2 on 1 plot -- use hold on
>> plot(t,y1);
>> hold on
>> plot(t,y2, 'r')
>> xlabel('time')
>> ylabel('value')
>> legend('sin', 'cos')
>> title('myplot')
>> print -dpng 'myPlot.png'
>> close

桙>> figure(1); plot(t,y1);
>> figure(2); plot(t,y2, 'r');
>> subplot(1,2,1); % divides plot in 1 1 * 2 grid and accesses the first element
>> plot(t,y1)
>> subplot(1,2,2)
>> plot(t,y2, 'r')
>> axis([0.5 1 -1 1])

>> clf
>> % clears figure
>> A=magic(5)
A =

   17   24    1    8   15
   23    5    7   14   16
    4    6   13   20   22
   10   12   19   21    3
   11   18   25    2    9

>> imagesc(A)
>> imagesc(A), colorbar, colormap gray;
>> % it used comma chaining
>> a=1, b=2, c=3
a =  1
b =  2
c =  3
>> % when using ; then is doesn't show results
>> a=1;b=2;

>> v=zeros(10,1)
v =

   0
   0
   0
   0
   0
   0
   0
   0
   0
   0

>> for i=1:10,
v(i) = 2^i;
end;
>> v
v =

      2
      4
      8
     16
     32
     64
    128
    256
    512
   1024
   
>> while i <=5,
v(i) = 100;
i=i+1;
end;

i=1
while true,
  v(i)=999;
  i=i+1;
  if i==6,
    break;
  end;
 end

 
v(1)=2;
if v(1)==1,
  disp('val is one');
elseif v(1) == 2,
  disp('val is two');
else
  disp('val <> 1 or 2');
end;

% functions are in a file
>> squareThisNumber(3)
ans =  9
>> % Octave searchpath (advanced optional)
>> % use addpath('C:\users\ ... enz...') then octave searches for functions is alternative dirs!

squareAndCubeThisNumber(10)
[a,b] = squareAndCubeThisNumber(4)

X = [ 1 1; 1 2; 1 3]
y = [1;2;3]

theta = [0;1]

j = costFunctionJ(X, y, theta)

% gives zero as the dataset matches the predictions perfect
% lets now make the predictions not match the data
theta = [0;0];
j = costFunctionJ(X, y, theta)

% Vectorisation

