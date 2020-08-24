@testset "argmin_x" begin

#test argmin_x.jl for CSC sparse matrix format
A=sprandn(100,100,0.01)+sparse(I, 100, 100);
A=A'*A;
while rank(Array(A))<100
  A=A+sparse(I, 100, 100)
end
xt=randn(100)
b=A*xt;

maxit=100;
pp=2
p=2
TF=Float64
log_PARSDMM = log_type_PARSDMM(zeros(maxit,pp),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),zeros(maxit),
zeros(maxit),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),TF(0),TF(0),TF(0),TF(0),TF(0),TF(0),TF(0));


#test for zero initial guess
x_min_solver="CG_normal"
(x,iter,relres)=argmin_x(A,b,zeros(100),x_min_solver,1e-5,1,log_PARSDMM)
@test norm(A*x-b)/norm(b) <= 1e-5
@test isapprox(norm(A*x-b)/norm(b),relres,rtol=1e-5)

#test for good initial guess
x_min_solver="CG_normal"
(x2,iter2,relres2)=argmin_x(A,b,xt+randn(100).*1e-4,x_min_solver,1e-5,1,log_PARSDMM)
@test iter2<iter
@test norm(A*x2-b)/norm(b) <= 1e-5
@test isapprox(norm(A*x2-b)/norm(b),relres2,rtol=1e-5)

#test if 10*eps(Float32/Float64) is achieved
x_min_solver="CG_normal"
(x,iter,relres)=argmin_x(A,b,zeros(100),x_min_solver,10*eps(),1,log_PARSDMM)
@test norm(A*x-b)/norm(b) <= 20*eps()

#test argmin_x.jl for CDS sparse matrix format
A=sprandn(100,100,0.01)+sparse(I, 100, 100);
A=A'*A;
while rank(Array(A))<100
  A=A+sparse(I, 100, 100)
end
xt=randn(100)
b=A*xt;

(R,offset) = mat2CDS(A)

x_min_solver="CG_normal"
y=zeros(Float64,length(b))
(x,iter,relres)=argmin_x(A,b,zeros(100),x_min_solver,1e-5,1,log_PARSDMM,offset,y)

@test norm(A*x-b)/norm(b) <= 1e-5
@test norm(A*x-b)/norm(b) <= 2.0*relres

#test for more accurate solutions with CDS
y=zeros(Float64,length(b))
(x,iter,relres)=argmin_x(A,b,zeros(100),x_min_solver,1e-10,1,log_PARSDMM,offset,y)

@test norm(A*x-b)/norm(b) <= 1e-10
@test norm(A*x-b)/norm(b) <= 2.0*relres

end
