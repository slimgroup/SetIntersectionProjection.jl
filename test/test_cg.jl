@testset "cg" begin

#test cg.jl

#positive semidefinite matrix
A=randn(200,100);  A=A'*A;
xt=randn(100)
b=A*xt

#inexact
(x,flag,relres,iter1) = cg(A,b,tol=1e-5,maxIter=1000,out=0);
@test norm(A*x-b)/norm(b) <= 1e-5

#test if very accurate solution can be achieved
(x,flag,relres,iter) = cg(A,b,tol=1e-14,maxIter=1000,out=0);
@test norm(A*x-b)/norm(b) <= 1.01e-14

#test with very good initial guess
x=deepcopy(xt) .+ eps();
(x,flag,relres,iter2) = cg(A,b,tol=1e-5,maxIter=1000,x=x,out=0);
@test norm(A*x-b)/norm(b) <= 1e-5
@test iter2<iter1 #test if the very good initial guess reduced the number of iterations

#test if the output is the input when the initial guess is the true solution
x=deepcopy(xt);
(x,flag,relres,iter2) = cg(A,b,tol=1e-14,maxIter=1000,x=x,out=0);
@test norm(A*x-b)/norm(b) <= 1e-14
@test iter2==1
@test x==xt

#test if simple preconditioner reduces the number of iterations
A=diagm(0 => (1:1:100))
(x1,flag,relres1,iter1) = cg(A,b,tol=1e-12,maxIter=1000,out=0); #unpreconditioned cg
Prec(input) = input./diag(A); #Jacobi preconditioner
(x2,flag,relres2,iter2) = cg(A,b,tol=1e-12,maxIter=1000,M=Prec,out=0);
@test iter2<iter1

end
