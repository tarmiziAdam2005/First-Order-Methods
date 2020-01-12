
function out = ista_19(y,A,AT,opts)

% Created on 16/12/2019 by Tarmizi Adam
%          Updates: 23/11/2019

% Iterative Shrinkage and Thresholding Algorithm (ISTA) for solving the
% following problem (Basis pursuit denoising):
%
%        minimize_x = 1/2 || Ax - y ||_2^2 + lam||x||_1,
%
% from the following forward model:
%
%              y = Ax + noise
%
% where A: is a randomized matrix where the columns are normalized.
%       x: is the sparse signal to be recovered.
%       y: is the noisy observed signal.

%Initializations

Nit = opts.Nit;
t   = opts.t;
L   = opts.L;
m   = opts.sigLen;
tol = opts.tol;
relError = zeros(m,1);

xk = y; %Our initial x i.e., x_0;

%% Main ISTA loop

for k=1:Nit
    
    x_old = xk; % Keep track of our previous x.
    
    Ax = A(xk);
    
    v = xk - (1/L)*AT(Ax - y); % The gradient step.
     
   % xk = shrink(v, 1/2*t); % The proximal step.
    xk = shrink(v, t);
    
    Err = xk -  x_old;
    relError(k) = norm(Err,'fro')/norm(xk,'fro');
    
    if relError(k) < tol
        break;
    end    
    
end

%% Otput of ISTA algorithm

out.sol = xk;
out.err = relError(1:k);

end

% Soft thresholding i.e., the proximal mapping for the l1-norm.
function z = shrink(x,r)
z = sign(x).*max(abs(x)-r,0);
end


