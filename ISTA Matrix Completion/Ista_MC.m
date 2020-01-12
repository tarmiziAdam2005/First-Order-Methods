function out = Ista_MC(Y,X,P,opts)


Nit = opts.Nit;
t   = opts.t;
L   = opts.L;
tol = opts.tol;
relError = zeros(Nit,1);
Xk = zeros(size(X));

for k =1:Nit
    
    X_old = Xk;
    
    V  = Xk - (1/L)*(P.*(Xk-Y));
    
    Xk = svt(V,t);
    
    
    Err = Xk -  X_old;
    relError(k) = norm(Err,'fro')/norm(Xk,'fro');
    
    if relError(k) < tol
        break;
    end 
    
end


out.sol = Xk;
out.err = relError(1:k);


end

function Z = shrink(X,r)  %Shrinkage operator
    Z = sign(X).*max(abs(X)- r,0);
end

function Z = svt(X, r) %Singular value thresholding

    [U, S, V] = svd(X);
    s = shrink(S,r);
    
    Z = U*s*V';
end





