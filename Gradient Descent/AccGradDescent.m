function out = AccGradDescent(y,A,AT,opts)

%Created by Tarmizi Adam (20/1/2020)
% Nesterov Accelerated Gradient descent (GD) algorithm with constant step size.
% Step size is chosen to be 1/L, where L is the Lipschitz constant.
% For problem (quadratic least squares problem):
%
%            Minimize F(x) = 1/2 || Ax - y||^2_2

Nit = opts.Nit;
L   = opts.L;
m   = opts.sigLen;
tol = opts.tol;
relError = zeros(Nit,1);
funcVal = zeros(Nit,1);

x_0  = 0.5*randn([m 1]); %Our initial x i.e., x_0; 
xk = x_0;
tk = 1;
yk = xk;
%% Main GD loop

for k=1:Nit
    
    x_old = xk; % Keep track of our previous x.
    
    Ayk = A(yk);
    
    xk = yk - (1/L)*AT(Ayk - y); % The gradient step.
    
    % **** Nesterov accelerated interpolation scheme ****
    t_old = tk;
    tk =(1 + sqrt(1 + 4*(t_old)^2))/2;
    yk = xk + ((t_old - 1)/(tk))*(xk -x_old);
    
    
    Err = xk -  x_old;
    relError(k) = norm(Err,'fro')/norm(xk,'fro');
    funcVal(k) = 0.5*norm(Ayk - y)^2;
    
    if relError(k) < tol
        break;
    end    
    
end

%% Otput of GD algorithm

out.sol = xk;
out.err = relError(1:k);
out.objVal = funcVal(1:k);
end