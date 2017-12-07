@testset "test_rhs_compose" begin
#test rhs_compose.jl

TF=Float64
TI=Int64

#with BLAS serial
rhs=randn(100000)
rhs_2 = deepcopy(rhs)
rhs_3 = deepcopy(rhs)
p=2
i=10


y=Vector{Vector{Float64}}(p)
y[1]=randn(51000)
y[2]=randn(100000)

l=Vector{Vector{Float64}}(p)
l[1]=randn(51000)
l[2]=randn(100000)

rho=[1.234;10.23432]

TD_OP = Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joLinearFunction{TF,TF}}}(p)
TD_OP[1] = speye(51000,100000)*2.0
TD_OP[2] = speye(100000)

parallel=false
Blas_active = false
rhs_compose!(rhs,l,y,rho,TD_OP,p,Blas_active,parallel)

#with explicit BLAS callse, serial Julia
Blas_active = true
rhs_compose!(rhs_2,l,y,rho,TD_OP,p,Blas_active,parallel)

@test isapprox(rhs,rhs_2,rtol=10*eps(TF))

#test parallel version
parallel=true

#distributed quantities
TD_OP_d  = distribute(TD_OP)
rho_d    = distribute(rho)
y_d      = distribute(y)
l_d      = distribute(l)

rhs_3=rhs_compose!(rhs_3,l_d,y_d,rho_d,TD_OP_d,p,Blas_active,parallel)

@test isapprox(rhs,rhs_3,rtol=10*eps(TF))

end
