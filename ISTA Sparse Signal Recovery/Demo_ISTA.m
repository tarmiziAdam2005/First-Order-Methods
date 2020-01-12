
% Created by Tarmizi Adam, 23/11/2019
% Simple demo of the the generalized gradient (proximal gradient method)
% method to solve the Basis Pursuit Denoising (BPDN). This algorithm is 
% well known as iterative shrinkage and thresholding algorithm (ISTA)

% For more details of the algorithm, please refer to the accompanying 
% file "ista_19.m" for details

% This demo file calls the main solver "ista_19()" to solve the BPDN problem.

clc;
clear all;
close all;

%Create a sparse signal;

N = 100; % Total length of sparse signal;
n = 20; % number of non-zero elements;
x = zeros(N,1);
r = -5 + (5+5)*rand(n,1); %random signal amplitude between -5 and 5
k = randi([0 100],1,n); % locations of the sparse singal
x(k) = r; % The sparse signal

%x = full(sprandn(m,1,0.1));
%load('x.mat');

if size(x,1) ==1
    x = x';
end

m = size(x,1);

%Create a random normalized sensing matrix A

A = randn(m,m);
d = sqrt(sum(A.^2));
A = A*sparse(1:m,1:m,1./d);

% function handles to the sensing matrix A, AT ( A transposed)
Afun = @(x) A*x;  
ATfun = @(x) A'*x;


%% Add the noise to the signal
sigma =0.3;

noise = sigma*randn(m,1);

y = Afun(x) + noise; % Add some noise to the signal.


%opts.relError = zeros(m,1);

lam = 0.5; %regularization parameter
opts.L = max(max(eig(A'*A))); % Lipschitz constant

opts.t = lam/opts.L; % Step size
opts.Nit = 100; % Number of iteration for algorithm termination
opts.tol = 1e-3;
opts.sigLen = m;
%xk = y;


out = ista_19(y,Afun,ATfun,opts);


subplot(4,1,1);
axis tight
stem(x);
title('Original sparse signal x');

subplot(4,1,2)
stem(out.sol)
title('Recovered sparse signal by ISTA');

subplot(4,1,3);
plot(y);
title('Observed noisy signal');

subplot(4,1,4);
plot(out.err);
title('Relative error');


