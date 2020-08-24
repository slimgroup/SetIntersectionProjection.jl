@testset "update_y_l" begin
#test update_y_l.jl

TF=Float64
TI=Int64

#without BLAS
x=randn(100)
p=2
i=10
Blas_active = false

y=Vector{Vector{Float64}}(undef, p)
y[1]=randn(51)
y[2]=randn(100)
y_old=Vector{Vector{Float64}}(undef, p)
y_old[1]=randn(51)
y_old[2]=randn(100)
l_old=Vector{Vector{Float64}}(undef, p)
l_old[1]=randn(51)
l_old[2]=randn(100)
l=Vector{Vector{Float64}}(undef, p)
l[1]=randn(51)
l[2]=randn(100)

rho=[1.234;10.23432]
gamma=Vector{Float64}(undef, p)
gamma[1]=1.0
gamma[2]=1.345

prox=Vector{Any}(undef, p)
prox[1] = x -> 1.0.*x
m=randn(100)
prox[2] = x -> prox_l2s!(x,rho[2],m)
TD_OP = Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joLinearFunction{TF,TF}}}(undef, p)
TD_OP[1] = sparse(1.0*I, 51,100)*2.0
TD_OP[2] = sparse(1.0*I, 100, 100)
maxit=39
log_PSDMM = log_type_PARSDMM(zeros(maxit,p-1),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),zeros(maxit),zeros(maxit),zeros(maxit,p),
zeros(maxit,p),zeros(maxit),zeros(maxit),0.00,0.00,0.00,0.00,0.00,0.00,0.00);

P_sub = Vector{Any}(undef, 1)
P_sub[1] = prox[1]
counter=12
x_hat=Vector{Vector{Float64}}(undef, p)
x_hat[1]=randn(51)
x_hat[2]=randn(100)
r_pri=Vector{Vector{Float64}}(undef, p)
r_pri[1]=randn(51)
r_pri[2]=randn(100)
s=Vector{Vector{Float64}}(undef, p)
s[1]=randn(51)
s[2]=randn(100)

#construct rerefrence solution
x_hat2=deepcopy(x_hat)
y2=deepcopy(y)
l2=deepcopy(l)
x2=deepcopy(x)
for i=1:p
  x_hat2[i]=gamma[i]*TD_OP[i]*x2 + (1-gamma[i])*y2[i]
  y2[i]=prox[i](x_hat2[i]-l2[i]./rho[i])
  l2[i].=l2[i].+rho[i].*(y2[i].-x_hat2[i])
end

(y,l,r_pri,s,log_PSDMM,counter,y_old,l_old)=update_y_l(x,p,i,Blas_active,
  y,y_old,l,l_old,rho,gamma,prox,TD_OP,log_PSDMM,P_sub,counter,x_hat,r_pri,s)

for i=1:p #don't test relative norm here, because l may be zeros/eps() valued
  @test norm(y[i]-y2[i]) <=1e-14
  @test norm(l[i]-l2[i]) <=1e-14
  @test norm(s[i]-TD_OP[i]*x) <= 1e-14
  @test norm(r_pri[i]-(-TD_OP[i]*x+y[i])) <=1e-14
end

#check that gamma and rho do not change
@test gamma[2]==1.345
@test gamma[1]==1.0
@test rho[1]==1.234
@test rho[2]==10.23432

#verify that y_old, l_old are not the same as current y and l
for i=1:p #don't test relative norm here, because l may be zeros/eps() valued
  @test norm(y[i]-y_old[i]) >10*eps()
  @test norm(l[i]-l_old[i]) >10*eps()
end

#now with BLAS
x=randn(100)
p=2
i=10
Blas_active = true

y=Vector{Vector{Float64}}(undef, p)
y[1]=randn(51)
y[2]=randn(100)
y_old=Vector{Vector{Float64}}(undef, p)
y_old[1]=randn(51)
y_old[2]=randn(100)
l_old=Vector{Vector{Float64}}(undef, p)
l_old[1]=randn(51)
l_old[2]=randn(100)
l=Vector{Vector{Float64}}(undef, p)
l[1]=randn(51)
l[2]=randn(100)

rho=[1.234;10.23432]
gamma=Vector{Float64}(undef, p)
gamma[1]=1.0
gamma[2]=1.345

prox=Vector{Any}(undef, p)
prox[1] = x -> 1.0.*x
m=randn(100)
prox[2] = x -> prox_l2s!(x,rho[2],m)
TD_OP = Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joLinearFunction{TF,TF}}}(undef, p)

TD_OP[1] = sparse(1.0*I, 51,100)*2.0
TD_OP[2] = sparse(1.0*I, 100, 100)
maxit=39
log_PSDMM = log_type_PARSDMM(zeros(maxit,p-1),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),zeros(maxit),zeros(maxit),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),0.00,0.00,0.00,0.00,0.00,0.00,0.00);
P_sub = Vector{Any}(undef, 1)
P_sub[1] = prox[1]
counter=12
x_hat=Vector{Vector{Float64}}(undef, p)
x_hat[1]=randn(51)
x_hat[2]=randn(100)
r_pri=Vector{Vector{Float64}}(undef, p)
r_pri[1]=randn(51)
r_pri[2]=randn(100)
s=Vector{Vector{Float64}}(undef, p)
s[1]=randn(51)
s[2]=randn(100)

#construct rerefrence solution
x_hat2=deepcopy(x_hat)
y2=deepcopy(y)
l2=deepcopy(l)
x2=deepcopy(x)
for i=1:p
  x_hat2[i]=gamma[i]*TD_OP[i]*x2 + (1-gamma[i])*y2[i]
  y2[i]=prox[i](x_hat2[i]-l2[i]./rho[i])
  l2[i].=l2[i].+rho[i].*(y2[i].-x_hat2[i])
end

(y,l,r_pri,s,log_PSDMM,counter,y_old,l_old)=update_y_l(x,p,i,Blas_active,
y,y_old,l,l_old,rho,gamma,prox,TD_OP,log_PSDMM,P_sub,counter,x_hat,r_pri,s)

for i=1:p #don't test relative norm here, because l may be zeros/eps() valued
  @test norm(y[i]-y2[i]) <=1e-14
  @test norm(l[i]-l2[i]) <=1e-14
  @test norm(s[i]-TD_OP[i]*x) <= 1e-14
  @test norm(r_pri[i]-(-TD_OP[i]*x+y[i])) <=1e-14
end

#check that gamma and rho do not change
@test gamma[2]==1.345
@test gamma[1]==1.0
@test rho[1]==1.234
@test rho[2]==10.23432

#verify that y_old, l_old are not the same as current y and l
for i=1:p #don't test relative norm here, because l may be zeros/eps() valued
  @test norm(y[i]-y_old[i]) >10*eps()
  @test norm(l[i]-l_old[i]) >10*eps()
end

end
