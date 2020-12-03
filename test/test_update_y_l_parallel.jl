@testset "update_y_l_parallel" begin
#test update_y_l_parallel.jl

TF=Float64
TI=Int64

#without BLAS
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

set_feas=zeros(length(TD_OP))

#distributed quantities
s_d      = distribute(s)
r_pri_d  = distribute(r_pri)
x_hat_d  = distribute(x_hat)
prox_d   = distribute(prox)
P_sub_d  = distribute(P_sub)
TD_OP_d  = distribute(TD_OP)
rho_d    = distribute(rho)
gamma_d  = distribute(gamma)
y_d      = distribute(y)
y_old_d  = distribute(y_old)
l_d      = distribute(l)
l_old_d  = distribute(l_old)
set_feas_d=distribute(set_feas)

#distributed computation
[ @spawnat pid  update_y_l_parallel(x,i,Blas_active,
  y_d[:L],y_old_d[:L],l_d[:L],l_old_d[:L],rho_d[:L],gamma_d[:L],prox_d[:L],TD_OP_d[:L]
  ,P_sub_d[:L],x_hat_d[:L],r_pri_d[:L],s_d[:L],set_feas_d[:L]) for pid in y_d.pids]

  rhs_next_iter=zeros(length(x))
  rhs_next_iter=@distributed (+) for i=1:2
      #rhs_d[i]
      TD_OP_d[i]'*(l_d[i].+y_d[i])
  end

# (y,l,r_pri,s,log_PSDMM,counter,y_old,l_old)=update_y_l_parallel(x,p,i,Blas_active,
#   y_d[:L],y_old_d[:L],l_d[:L],l_old_d[:L],rho_d[:L],gamma_d[:L],prox_d[:L],TD_OP_d[:L],
#   log_PSDMM,P_sub_d[:L],counter,x_hat_d[:L],r_pri_d[:L],s_d[:L])


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
rhs_next_iter_ref=zeros(length(x))
for i  = 1:p
    #rhs_d[i]
    rhs_next_iter_ref=rhs_next_iter_ref+TD_OP[i]'*(l2[i].+y2[i])
end


#use serial update_y_l.jl
(y,l,r_pri,s,log_PSDMM,counter,y_old,l_old)=update_y_l(x,p,i,Blas_active,
  y,y_old,l,l_old,rho,gamma,prox,TD_OP,log_PSDMM,P_sub,counter,x_hat,r_pri,s)
  rhs_next_iter_serial=zeros(length(x))
  for i  = 1:p
      #rhs_d[i]
      rhs_next_iter_serial=rhs_next_iter_serial+TD_OP[i]'*(l[i].+y[i])
  end

@test isapprox(rhs_next_iter_serial,rhs_next_iter_ref,rtol=10*eps())
@test isapprox(rhs_next_iter,rhs_next_iter_ref,rtol=10*eps())
@test isapprox(s,s_d,rtol=10*eps())
@test isapprox(l,l_d,rtol=10*eps())
@test isapprox(r_pri,r_pri_d,rtol=10*eps())

end
