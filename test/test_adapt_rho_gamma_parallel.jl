@testset "adapt_rho_gamma_parallel" begin
#test adapt_rho_gamma_parallel.jl
#test if the parallel and serial version yield the same results

#without BLAS
x=randn(100)
p=2
i=10


y=Vector{Vector{Float64}}(p)
y[1]=randn(51)
y[2]=randn(100)
y_0=Vector{Vector{Float64}}(p)
y_0[1]=randn(51)
y_0[2]=randn(100)
y_old=Vector{Vector{Float64}}(p)
y_old[1]=randn(51)
y_old[2]=randn(100)
l_old=Vector{Vector{Float64}}(p)
l_old[1]=randn(51)
l_old[2]=randn(100)
l=Vector{Vector{Float64}}(p)
l[1]=randn(51)
l[2]=randn(100)
d_l=Vector{Vector{Float64}}(p)
d_l[1]=randn(51)
d_l[2]=randn(100)
l_0=Vector{Vector{Float64}}(p)
l_0[1]=randn(51)
l_0[2]=randn(100)
l_hat=Vector{Vector{Float64}}(p)
l_hat[1]=randn(51)
l_hat[2]=randn(100)
l_hat_0=Vector{Vector{Float64}}(p)
l_hat_0[1]=randn(51)
l_hat_0[2]=randn(100)
d_l_hat=Vector{Vector{Float64}}(p)
d_l_hat[1]=randn(51)
d_l_hat[2]=randn(100)
d_H_hat=Vector{Vector{Float64}}(p)
d_H_hat[1]=randn(51)
d_H_hat[2]=randn(100)
d_G_hat=Vector{Vector{Float64}}(p)
d_G_hat[1]=randn(51)
d_G_hat[2]=randn(100)

rho=[1.234;10.23432]
gamma=Vector{Float64}(p)
gamma[1]=1.0
gamma[2]=1.345

prox=Vector{Any}(p)
prox[1] = x -> 1.0.*x
m=randn(100)
prox[2] = x -> prox_l2s!(x,rho[2],m)
TD_OP=Vector{SparseMatrixCSC{Float64,Int64}}(p)
TD_OP[1] = speye(51,100)*2.0
TD_OP[2] = speye(100)
maxit=39
log_PSDMM = log_type_PARSDMM(zeros(maxit,p-1),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),zeros(maxit),zeros(maxit),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),0.00,0.00,0.00,0.00,0.00,0.00,0.00);
P_sub = Vector{Any}(1)
P_sub[1] = prox[1]
counter=12
x_hat=Vector{Vector{Float64}}(p)
x_hat[1]=randn(51)
x_hat[2]=randn(100)
r_pri=Vector{Vector{Float64}}(p)
r_pri[1]=randn(51)
r_pri[2]=randn(100)
s=Vector{Vector{Float64}}(p)
s[1]=randn(51)
s[2]=randn(100)
s_0=Vector{Vector{Float64}}(p)
s_0[1]=randn(51)
s_0[2]=randn(100)

#distributed quantities
s_d      = distribute(s)
s_0_d    = distribute(s_0)
r_pri_d  = distribute(r_pri)
x_hat_d  = distribute(x_hat)
prox_d   = distribute(prox)
P_sub_d  = distribute(P_sub)
TD_OP_d  = distribute(TD_OP)
rho_d    = distribute(rho)
gamma_d  = distribute(gamma)
y_d      = distribute(y)
y_0_d    = distribute(y_0)
y_old_d  = distribute(y_old)
l_d      = distribute(l)
d_l_d    = distribute(d_l)
l_old_d  = distribute(l_old)
l_0_d    = distribute(l_0)
l_hat_d  = distribute(l_hat)
d_l_hat_d= distribute(d_l_hat)
d_H_hat_d= distribute(d_H_hat)
d_G_hat_d=distribute(d_G_hat)
l_hat_0_d=distribute(l_hat_0)

#distributed computation
[ @spawnat pid  adapt_rho_gamma_parallel(gamma_d[:L],rho_d[:L],true,true,"BB",y_d[:L],y_old_d[:L],s_d[:L],s_0_d[:L],l_d[:L],l_hat_0_d[:L],
                                         l_0_d[:L],l_old_d[:L],y_0_d[:L],l_hat_d[:L],d_l_hat_d[:L],d_H_hat_d[:L],d_l_d[:L],d_G_hat_d[:L]) for pid in y_d.pids]


#serial computation
(rho,gamma,l_hat,d_l_hat,d_H_hat,d_l,d_G_hat)=adapt_rho_gamma(i,gamma,rho,true,true,"BB",y,y_old,s,s_0,l,l_hat_0,l_0,l_old,y_0,p,l_hat,d_l_hat,d_H_hat,d_l,d_G_hat);

#see if output is the same
@test rho_d==rho
@test gamma==gamma_d
@test isapprox(l_hat,l_hat_d,rtol=10*eps())
@test isapprox(d_l_hat,d_l_hat_d,rtol=10*eps())
@test isapprox(d_H_hat,d_H_hat_d,rtol=10*eps())
@test isapprox(d_l,d_l_d,rtol=10*eps())
@test isapprox(d_G_hat,d_G_hat_d,rtol=10*eps())

#test another related piece of code
for ii = 1:p
  l_hat[ii]   = l_old[ii] .+ rho[ii].* ( -s[ii] .+ y_old[ii] );
  copy!(l_hat_0[ii],l_hat[ii])
  copy!(y_0[ii],y[ii])
  copy!(s_0[ii],s[ii])
  copy!(l_0[ii],l[ii])
end

[@spawnat pid l_hat_d[:L] = l_old_d[:L] .+ rho_d[:L].* ( -s_d[:L] .+ y_old_d[:L] ) for pid in l_hat_d.pids]
copy!(l_hat_0_d,l_hat_d)
copy!(y_0_d,y_d)
copy!(s_0_d,s_d)
copy!(l_0_d,l_d)

@test isapprox(l_hat_0_d,l_hat_0,rtol=10*eps())
@test isapprox(l_hat,l_hat_d,rtol=10*eps())
@test isapprox(y_0_d,y_d,rtol=10*eps())
@test isapprox(s_0_d,s_d,rtol=10*eps())
@test isapprox(l_0_d,l_d,rtol=10*eps())

end
