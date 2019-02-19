@testset "setup_constraints" begin

FL=64
TF=Float64

comp_grid = compgrid((1.0, 1.0),(100, 201))
comp_grid3D = compgrid((1.0, 1.0, 1.0),(50, 60, 30))

options    = PARSDMM_options()
options.FL = TF

#test project_bounds! with scalar bounds
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = -0.11
  m_max     = 0.01
  set_type  = "bounds"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)

  x = randn(100)
  P_sub[1](x)
  @test maximum(x) <= m_max
  @test minimum(x) >= m_min

#test project_bounds! with vector bounds
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = randn(100) .- 10.0
  m_max     = randn(100) .+ 10.0
  set_type  = "bounds"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)

  x = 100.0 .* randn(100)
  P_sub[1](x)
  [@test x[i] <= m_max[i] for i=1:100]
  [@test x[i] >= m_min[i] for i=1:100]


#test project l1_Duchi!
  #see if projector returns untouched vector
  x=randn(100)
  y=deepcopy(x)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = norm(x,1)*2
  set_type  = "l1"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](x)
  @test x == y

  #choose smaller constraint than vector l1 norm
  x=randn(100)
  y=deepcopy(x)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = norm(x,1)*0.234
  set_type  = "l1"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](x)
  @test isapprox(norm(x,1),m_max,rtol=10*eps())


#test project_cardinality!

  #test vector mode
  x=randn(100)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 5
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](x)
  @test count(!iszero, x)==m_max

  #test on closed form solution (vector)
  x = [0;0;-1;2;-3]
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 2
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](x)
  @test x==[0;0;0;2;-3]

  #test mode that projects each column of a matrix separately
  X = randn(comp_grid.n)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 7
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("fiber","x")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](vec(X))
  X = reshape(X,comp_grid.n)
  [@test count(!iszero, X[:,i]) == 7 for i=1:size(X,2)]

  #test mode that projects each row of a matrix separately
  X = randn(comp_grid.n)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 11
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("fiber","z")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](vec(X))
  X = reshape(X,comp_grid.n)
  [@test count(!iszero, X[i,:]) == 11 for i=1:size(X,1)]

  #test mode that projects each fiber of a tensor separately (either in x,y or z direction)
  X = randn(comp_grid3D.n)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 7
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("fiber","x")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](vec(X))
  X = reshape(X,comp_grid3D.n)
  [@test count(!iszero, X[:,i,j]) == 7 for i=1:size(X,2) for j=1:size(X,3)]

  X = randn(comp_grid3D.n)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 6
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("fiber","y")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](vec(X))
  X = reshape(X,comp_grid3D.n)
  [@test count(!iszero, X[i,:,j]) == 6 for i=1:size(X,1) for j=1:size(X,3)]

  X = randn(comp_grid3D.n)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 4
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("fiber","z")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](vec(X))
  X = reshape(X,comp_grid3D.n)
  [@test count(!iszero, X[i,j,:]) == 4 for i=1:size(X,1) for j=1:size(X,2)]

  #test mode that projects onto each slice of a tensor separately (either in x,y or z direction)
  X = vec(randn(comp_grid3D.n))
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 7
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("slice","x")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  X = P_sub[1](X)
  X = reshape(X,comp_grid3D.n)
  [@test count(!iszero, X[i,:,:]) == 7 for i=1:comp_grid3D.n[1]]

  X = vec(randn(comp_grid3D.n))
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 6
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("slice","y")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  X = P_sub[1](X)
  X = reshape(X,comp_grid3D.n)
  [@test count(!iszero, X[:,i,:]) == 6 for i=1:comp_grid3D.n[2]]

  X = vec(randn(comp_grid3D.n))
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 5
  set_type  = "cardinality"
  TD_OP     = "identity"
  app_mode  = ("slice","z")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  X = P_sub[1](X)
  X = reshape(X,comp_grid3D.n)
  [@test count(!iszero, X[:,:,i]) == 5 for i=1:comp_grid3D.n[3]]


# test project_l2!
  #test for sigma smaller than 2-norm of vector
  x=randn(100)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 0.123
  set_type  = "l2"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](x)
  @test isapprox(norm(x,2),0.123,rtol=eps()*10)

  #test for sigma larger than 2-norm of vector
  x=randn(100)
  y=deepcopy(x)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 1.234*norm(x,2)
  set_type  = "l2"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](x)
  @test x==y


#test project_rank! (codes assumes input and output are vectors)

  # test if input is returned untouched if input matrix rank is less than the constraint
  X = vec(randn(comp_grid.n))
  XX = deepcopy(X)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = minimum(comp_grid.n)
  set_type  = "rank"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](X)
  @test isapprox(XX,X,rtol=20*eps())

  # test if output matrix has correct rank when constraint is smaller than matrix rank
  X = vec(randn(comp_grid.n))
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 12
  set_type  = "rank"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](X)
  X=reshape(X,comp_grid.n)
  r_X=rank(X)
  @test r_X==12

  #test rank projection on slices of a 3D tensor
  X = vec(randn(comp_grid3D.n))
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 7
  set_type  = "rank"
  TD_OP     = "identity"
  app_mode  = ("slice","x")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](X)
  X=reshape(X,comp_grid3D.n)
  [@test rank(X[i,:,:]) == 7 for i=1:size(X,1)]

  X = vec(randn(comp_grid3D.n))
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 6
  set_type  = "rank"
  TD_OP     = "identity"
  app_mode  = ("slice","y")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](X)
  X=reshape(X,comp_grid3D.n)
  [@test rank(X[:,i,:]) == 6 for i=1:size(X,2)]

  X = vec(randn(comp_grid3D.n))
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 5
  set_type  = "rank"
  TD_OP     = "identity"
  app_mode  = ("slice","z")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](X)
  X=reshape(X,comp_grid3D.n)
  [@test rank(X[:,:,i]) == 5 for i=1:size(X,3)]


#test project nuclear! (codes assumes input and output are vectors)
  # test if input is returned untouched if nuclear norm is less than constraint
  X    = randn(comp_grid.n)
  nn_X = norm(svdvals(X),1)
  X    = vec(X)
  XX   = deepcopy(X)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = nn_X*1.1
  set_type  = "nuclear"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](X)
  @test isapprox(XX,X,rtol=20*eps())

  # test if output matrix has correct nuclear norm when constraint is smaller than matrix nuclear norm
  X    = randn(comp_grid.n)
  nn_X = norm(svdvals(X),1)
  X    = vec(X)
  XX   = deepcopy(X)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = nn_X*0.567
  set_type  = "nuclear"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](X)
  nn_Xp = norm(svdvals(reshape(X,comp_grid.n)),1)
  @test isapprox(nn_Xp,nn_X*0.567,rtol=20*eps())

  #test nuclear norm projection onto slices of a 3D tensor
  X = vec(randn(comp_grid3D.n))
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 1.234
  set_type  = "nuclear"
  TD_OP     = "identity"
  app_mode  = ("slice","x")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](X)
  X=reshape(X,comp_grid3D.n)
  [@test isapprox(norm(svdvals(X[i,:,:]),1), 1.234,rtol=100*eps()) for i=1:size(X,1)]

  X = vec(randn(comp_grid3D.n))
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 1.234
  set_type  = "nuclear"
  TD_OP     = "identity"
  app_mode  = ("slice","y")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](X)
  X=reshape(X,comp_grid3D.n)
  [@test isapprox(norm(svdvals(X[:,i,:]),1), 1.234,rtol=100*eps()) for i=1:size(X,2)]

  X = vec(randn(comp_grid3D.n))
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 1.234
  set_type  = "nuclear"
  TD_OP     = "identity"
  app_mode  = ("slice","z")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid3D,options.FL)
  P_sub[1](X)
  X=reshape(X,comp_grid3D.n)
  [@test isapprox(norm(svdvals(X[:,:,i]),1), 1.234,rtol=100*eps()) for i=1:size(X,3)]


#test project_subspace!
  #on a vector inputs
  #orthogonal subspace
  M=randn(100,50)
  F=svd(M)
  x=randn(100)
  y=deepcopy(x)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 0.0
  set_type  = "subspace"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = (F.U,true)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](x)
  @test isapprox(x,F.U*(F.U'*y),rtol=eps()*100)

  #non-orthogonal subspace
  M=randn(100,50)
  x=randn(100)
  y=deepcopy(x)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 0.0
  set_type  = "subspace"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = (M,false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](x)
  @test isapprox(x,M*((M'*M)\(M'*y)),rtol=eps()*10)

  #on matrix input: project every column onto the subspace
  M=randn(prod(comp_grid.n[1]),50)
  x=randn(prod(comp_grid.n))
  y=deepcopy(x)
  x=vec(x)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 0.0
  set_type  = "subspace"
  TD_OP     = "identity"
  app_mode  = ("fiber","x")
  custom_TD_OP = (M,false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](x)
  @test isapprox(x,vec(M*((M'*M)\(M'*reshape(y,comp_grid.n)))),rtol=eps()*10)

  #on matrix input: project every row onto the subspace
  M=randn(prod(comp_grid.n[2]),50)
  x=randn(prod(comp_grid.n))
  y=deepcopy(x)
  x=vec(x)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = 0
  m_max     = 0.0
  set_type  = "subspace"
  TD_OP     = "identity"
  app_mode  = ("fiber","z")
  custom_TD_OP = (M,false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](x)
  @test isapprox(x,vec((M*((M'*M)\(M'*reshape(y,comp_grid.n)')))'),rtol=eps()*10)

#test projection onto relaxed histogram
  #first test exact histogram projection
  ref = sort(randn(100))
  x   = randn(100)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = ref
  m_max     = ref
  set_type  = "histogram"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](x)
  @test ref == sort(x)

  #relaxed histogram projection
  x   = randn(100)
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  m_min     = sort(randn(100))
  m_max     = m_min.+0.7
  set_type  = "histogram"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  P_sub[1](x)
  x = sort(x)
  [@test x[i] <= m_max[i] for i=1:100]
  [@test x[i] >= m_min[i] for i=1:100]

# #test projection onto l1-ball with DFT operator
# constraint=Dict()
# constraint["use_TD_l1_2"]=true
# constraint["TD_l1_operator_2"]="DFT"
# constraint["TD_l1_sigma_2"] = 1.234
# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,FL)
#
# (TD_OP, dummy1, dummy2, dummy3)=get_TD_operator(comp_grid,"DFT",TF)
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
# constraint["TD_l1_operator_2"]="DFT"
# constraint["TD_l1_sigma_2"] = 10000000.0
# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,FL)
# x=randn(TF,100,201)
# x=vec(x)
# y=deepcopy(x)
# P_sub[1](x)
# @test isapprox(x,y,rtol=10*eps(TF))




end
