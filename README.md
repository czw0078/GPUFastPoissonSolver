# GPUFastPoissonSolver
A implement P3M method of long range force calculation, which is based on Fast Poisson Solver and is a critical component for MCGPU project.

1.Run the program in terminal
  
./a.out

2.Compile and run the source code file fastPoissonSolver.cu in the terminal:

nvcc fastPoissonSolver.cu -run -lcusparse -lcufft

a file named a.out will be generate
