clc;
clear all;
close all;

m = 100; %Size of matrix A.
conditionNumber = 100;
[A,y] = CreateA(conditionNumber,m); % This funtion creates a symmetric positive definite matrix

m = size(A,1);

Afun = @(x) A*x;  
ATfun = @(x) A'*x;

%Function handles for backtracking linesearc gradient descent...
gradF = @(x) A'*(A*x-y);
objF = @(x) 0.5*norm(A*x - y)^2;


opts.Nit = 1000; % Number of iteration for algorithm termination
opts.tol = 1e-5;
opts.sigLen = m;
opts.L = max(max(eig(A'*A))); % Lipschitz constant



out  = GradDesBackTrack(gradF, objF,opts);
out2 = GradDescent(y,Afun,ATfun,opts); %Run our gradient descent algorithm


optVal  = min(out.objVal);
optVal2 = min(out2.objVal);

subplot (3,1,1)
plot(out.err,'LineWidth',2.5, 'Color', 'green'); hold; 
plot(out2.err,'LineWidth',2.5, 'Color', 'blue'); 

xlabel('Iterations', 'FontSize',14);
ylabel('Relative error','FontSize',14);
legend('GD-BT','GD-Const');

subplot (3,1,2)
semilogy(out.err,'LineWidth',2.5, 'Color','green');hold;
semilogy(out2.err,'LineWidth',2.5, 'Color','blue');

xlabel('Iterations','FontSize',14);
legend('GD-BT','GD-Const');

subplot(3,1,3)
semilogy(out.objVal - optVal,'LineWidth',2.5, 'Color','green');hold;
semilogy(out2.objVal -optVal2,'LineWidth',2.5, 'Color','blue');

xlabel('Iterations', 'FontSize',14);
ylabel('F(x) - F*','FontSize',14);
legend('GD-BT','GD-Const');
