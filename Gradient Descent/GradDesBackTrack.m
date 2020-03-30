function out = GradDesBackTrack(gradF, objF,opts)

%Created by Tarmizi Adam (27/3/2020)
% Gradient descent (GD) algorithm with backtracking line search.
% Step size is chosen to by backtracking. Particularly usefull when we do
% not have the Lipschitz constant L, at hand. 

Nit = opts.Nit;
m   = opts.sigLen;
tol = opts.tol;
relError = zeros(m,1);

% Initialize the Backtracking constants 
%    0 < alpha <= 0.5
%    0 < beta < 1
s = 1;
beta = 0.9;
alpha =0.2;

xk = randn(m,1); %Our initial x i.e., x_0;
funcVal = zeros(Nit,1);

grad = gradF(xk);
obj = objF(xk); 
%% Main GD loop

for k=1:Nit
    
    x_old = xk; % Keep track of our previous x.
    t = s;
    
    while (obj - objF(xk-t*grad) < alpha*t*norm(grad)^2) 
        t = beta*t;
    end
     
    xk   = xk - t*grad; % The gradient step.
    obj  = objF(xk);
    funcVal(k) = obj;  % Store the function values
    grad = gradF(xk);
    
    Err = xk -  x_old;
    relError(k) = norm(Err,'fro')/norm(xk,'fro');
   
    if relError(k) < tol
        break;
    end    
    
end

%% Otput of GD algorithm

out.sol = xk;
out.err = relError(1:k);
out.objVal = funcVal(1:k);

end




