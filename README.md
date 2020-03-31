# First-Order-Methods
First order optimization methods

In this repository, I will add first order (gradient methods) algorithms. It will be updated from time to time.

Folders:

      Gradient descent
         - Vanilla gradient descent (solve least squares,LS problem)
         - Nesterov's accelerated gradient descent (LS problem).
         
      ISTA Matrix completion
         - Iterative shrinkage and thresholding algorithm (ISTA) for matrix completion. ISTA is a proximal gradient algorithm, thus a                first order algorithm.
         
      ISTA sparse signal recovery
         - ISTA for recovering a sparse signals (Basis pursuit denoising). Solves
                1/2||Ax -b||^2_2 + lambda||x||_1
