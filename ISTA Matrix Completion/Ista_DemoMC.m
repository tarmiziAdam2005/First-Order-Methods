clc;
clear all;
%close all;

%% For Toy example, use this %%
%{
n1 = 10;
n2 = 10;
r = 5;

X = rand(n1,r)*rand(r,n2); % Create a random n-by-n matrix.
                           % This Our original matrix
  %}                         
  %% For image inpainting %%
  
  X = imread('peppers.bmp');
  X = double(X);
  
  [n1,n2] =size(X);

%% Create projection matrix %%
J = randperm(n1*n2);
J = J(1:round(0.2*n1*n2)); %Change here for percentage of missing entries.
P = ones(n1*n2,1);
P(J) = 0;
P = reshape(P,[n1,n2]); % our projection matrix

%% Simulate our corrupted original matrix %%
Y = X(:);
sigma = 30; %noise level
noise = sigma*randn(n1*n2,1);

Y = Y + noise;
Y = reshape(Y,[n1,n2]);
Y = P.*Y; % Our final noisy + missing entry matrix (Observation)


%% Parameters for ISTA %%

lam = 800; %regularization parameter
opts.L = 1.1; % Lipschitz constant

opts.t = lam/opts.L; % Step size
opts.Nit = 500; % Number of iteration for algorithm termination
opts.tol = 1e-3;

out = Ista_MC(Y,X,P,opts);

figure;
imshow(uint8(Y));
figure;
imshow(uint8(out.sol));
figure;
imshow(uint8(X));

