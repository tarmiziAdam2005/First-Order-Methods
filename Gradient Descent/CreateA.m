function [A,y] = CreateA(CondNumb,n)
% Generate a dense n x n symmetric, positive definite matrix
% Code obtained from: 

% https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab

A = randn(n,n); % generate a random n x n matrix

% construct a symmetric matrix using either
%A = 0.5*(A+A'); %OR
%A = A*A';
% The first is significantly faster: O(n^2) compared to O(n^3)

% since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
%   is symmetric positive definite, which can be ensured by adding nI
%A = A + n*eye(n);

%{
r = 25;  % rank

U = randn([n r]);
[U,~] = qr(U);

V = randn([n r]);
[V, ~] = qr(V);

eig = linspace(CondNumb,1,min(n,n));

S = diag(eig);
A = U*S*V';

A = 0.5*(A+A');
%}

%
A = 0.5*(A+A');%Create matrix A, with desired condition number.
[U,S,V]=svd(A);
S(S~=0)=linspace(CondNumb,1,min(n,n));
A=U*S*V';


%}
y = randn(n,1);
%y = -A*b;


end
