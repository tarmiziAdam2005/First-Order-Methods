function out = GradDescent(y,A,AT,opts)

%Created by Tarmizi Adam (14/1/2020)
% Gradient descent (GD) algorithm with constant step size.
% Step size is chosen to be 1/L, where L is the Lipschitz constant.

Nit = opts.Nit;
L   = opts.L;
m   = opts.sigLen;
tol = opts.tol;
relError = zeros(m,1);

xk = y; %Our initial x i.e., x_0;

%% Main GD loop

for k=1:Nit
    
    x_old = xk; % Keep track of our previous x.
    
    Ax = A(xk);
    
    xk = xk - (1/L)*AT(Ax - y); % The gradient step.
    
    Err = xk -  x_old;
    relError(k) = norm(Err,'fro')/norm(xk,'fro');
    
    if relError(k) < tol
        break;
    end    
    
end

%% Otput of GD algorithm

out.sol = xk;
out.err = relError(1:k);

end