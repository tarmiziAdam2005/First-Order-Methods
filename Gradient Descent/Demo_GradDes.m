
% Created by Tarmizi Adam on 14/1/2020 (tarmizi_adam2005@yahoo.com)
% Demo to demonstrate the use of Gradient descent to solve a linear system
% or least square problem.

% We would like to solve the following least squares problem,

%              minimize F(x) = 0.5||Ax - y||_2,

% which is an unconstrained optimization problem. This problem is same as
% solving the linear system Ax = y. A is an m x m square matrix.

clc;
clear all;
close all;

m = 100; %Size of matrix A.

%Create a full column rank matrix A (all positive eigenvalues).
% However, when the size of A is big, the condition number becomes large.
% The solution to the linear system will therefore be wrong.
%{
X = 2*eye(m);
Y = diag(ones(m-1,1),-1);
Z = diag(ones(m-1,1),1);

A = X - Y - Z;
%}

%% Another option, create a singular matrix A with desired condition number.
%  This uses the singular value decomposition. The condition number is the
%  ratio between the maximum and minimum singular values.

nr=m;                   %Number of rows
nc=m;                   %Number of columns
CondNumb= 5;            %Desired condition number

A=randn(nr,nc);         %Create random  matrix.
[U,S,V]=svd(A);
S(S~=0)=linspace(CondNumb,1,min(nr,nc));
A=U*S*V';                %Create matrix A, with desired condition number.

%%

% function handles to the matrix A, AT ( A transposed)
Afun = @(x) A*x;  
ATfun = @(x) A'*x;

y  = -5 + (5+5)*rand(m,1); %random y between -5 and 5

opts.L = max(max(eig(A'*A))); % Lipschitz constant

opts.Nit = 1000; % Number of iteration for algorithm termination
opts.tol = 1e-10;
opts.sigLen = m;

out = GradDescent(y,Afun,ATfun,opts); %Run our gradient descent algorithm

% show results in command window.
x = out.sol

%% Some plottings

subplot(2,1,1)
axis tight
plot(out.err);
xlabel('GD Iterations','LineWidth',2);
ylabel('Relative error','LineWidth',2);

subplot(2,1,2)
axis tight
semilogy(out.err);
xlabel('GD Iterations');







