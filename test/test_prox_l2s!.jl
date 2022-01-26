@testset "prox_l2s!" begin
#test prox_ls2!

m=randn(10)
x=randn(10)
rho=0.0
prox_l2s!(x,rho,m)
@test x==m

rho = 1e10
y=deepcopy(x)
prox_l2s!(x,rho,m)
@test isapprox(x,y,rtol=1e-14)

x=Vector{Float64}(undef, 1); x[1]=2.0
m=Vector{Float64}(undef, 1); m[1]=1.0
rho=3.0
prox_l2s!(x,rho,m)
@test x[1]==(7/4)

end
