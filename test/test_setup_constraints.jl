@testset "setup_constraints" begin

FL=32
TF=Float32
type compgrid
  d :: Tuple
  n :: Tuple
end

comp_grid=compgrid((1.0, 1.0),(100, 201))

#test projection onto l1-ball with DFT operator
constraint=Dict()
constraint["use_TD_l1_2"]=true
constraint["TD_l1_operator_2"]="DFT"
constraint["TD_l1_sigma_2"] = 1.234
(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,FL)

(TD_OP, dummy1, dummy2, dummy3)=get_TD_operator(comp_grid,"DFT",TF)

#project vector with ||TD_OP*x||_1 > sigma norm
x=randn(TF,100,201)
x=vec(x)
P_sub[1](x)
@test isapprox(norm(TD_OP*x,1),1.234,rtol=100*eps(TF))

#project vector with ||TD_OP*x||_1 < sigma norm
constraint=Dict()
constraint["use_TD_l1_2"]=true
constraint["TD_l1_operator_2"]="DFT"
constraint["TD_l1_sigma_2"] = 10000000.0
(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,FL)
x=randn(TF,100,201)
x=vec(x)
y=deepcopy(x)
P_sub[1](x)
@test isapprox(x,y,rtol=10*eps(TF))

# #test projection onto l1-ball with curvelet operator
# constraint=Dict()
# constraint["use_TD_l1_2"]=true
# constraint["TD_l1_operator_2"]="curvelet"
# constraint["TD_l1_sigma_2"] = 1.234
# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,FL)
#
# (TD_OP, dummy1, dummy2, dummy3)=get_TD_operator(comp_grid,"curvelet",TF)
#
# #project vector with ||TD_OP*x||_1 > sigma norm
# x=randn(TF,100,201)
# x=vec(x)
# P_sub[1](x)
# @test isapprox(norm(TD_OP*x,1),1.234,rtol=100*eps(TF))
#
# #project vector with ||TD_OP*x||_1 < sigma norm
# constraint=Dict()
# constraint["use_TD_l1_2"]=true
# constraint["TD_l1_operator_2"]="curvelet"
# constraint["TD_l1_sigma_2"] = 10000000.0
# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,FL)
# x=randn(TF,100,201)
# x=vec(x)
# y=deepcopy(x)
# P_sub[1](x)
# @test isapprox(x,y,rtol=10*eps(TF))


end
