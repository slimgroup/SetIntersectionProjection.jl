@testset "parallel_PARSDMM" begin
Random.seed!(123)

#test parallel PARSDMM (test is for 2 workers)
    options = PARSDMM_options()
    default_PARSDMM_options(options,Float32)
    options.parallel     = true
    options.evol_rel_tol = 1e-5
    options.feas_tol     = 1e-5
    options.obj_tol      = 1e-5
    options.maxit        = 10000

    x         = randn(Float32,201,100)
    comp_grid = compgrid((1.0, 1.0),(201, 100))
    x         = vec(x)
    m         = deepcopy(x)

    constraint = Vector{SetIntersectionProjection.set_definitions}()

    #total variation
    (TD_OP, AtA_diag, dense, TD_n) = get_TD_operator(comp_grid,"TV",options.FL)
    m_min     = 0.0
    m_max     = 0.5f0*norm(TD_OP*x,1)
    set_type  = "l1"
    TD_OP     = "TV"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

    (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
    (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)
 
    (x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
    
    result = deepcopy(x1)
    for i=1:length(TD_OP)-1
      copy!(result,x1)
      @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= 1.5*options.feas_tol
    end

#test parallel PARSDMM without many explicit BLAS calls (test is for 2 workers)
    #default_PARSDMM_options(options,Float64)
    options.parallel    = true
    options.Blas_active = false

    (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
    (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)
 
    (x2,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
    result = deepcopy(x2)
    for i=1:length(TD_OP)-1
      copy!(result,x2)
      @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= 1.5*options.feas_tol
    end

#test if with and without expicit Blas calls results are the same
@test isapprox(x1,x2,rtol=1f-4)

#run serial PARSDMM and compare results
    options.parallel    = false

    (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
    (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

    (x3,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
    result = deepcopy(x3)
    for i=1:length(TD_OP)-1
      copy!(result,x3)
      @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= 1.5*options.feas_tol
    end

@test isapprox(x1,x3,rtol=5*1f-4)


#test parallel PARDMM with a JOLI operator in a projector, but not in TD_OP
  options = PARSDMM_options()
  default_PARSDMM_options(options,Float32)
  options.parallel     = false
  options.evol_rel_tol = 1e-5
  options.feas_tol     = 1e-5
  options.obj_tol      = 1e-5
  options.maxit        = 10000

  constraint = Vector{SetIntersectionProjection.set_definitions}()

  #DFT l1 constraints
  (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"DFT",options.FL)
  m_min     = 0.0
  m_max     = 0.5f0*norm(TD_OP*x,1)
  set_type  = "l1"
  TD_OP     = "DFT"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

  (x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

  result = deepcopy(x1)
  for i=1:length(TD_OP)-1
    copy!(result,x1)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= 1.5*options.feas_tol
  end

  #test in parallel and see if results are the same
  options.parallel     = true

  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

  (x2,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

  result = deepcopy(x2)
  for i=1:length(TD_OP)-1
    copy!(result,x2)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= 1.5*options.feas_tol
  end

  @test isapprox(x1,x2,rtol=5f-4)

# #test parallel PARDMM with a JOLI operator in TD_OP
# default_PARSDMM_options(options,Float64)
# options.parallel    = false

# constraint=Dict()

# #DFT l1 constraints
# (C, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"curvelet",options.FL)
# constraint["use_TD_l1_1"]      = true
# constraint["TD_l1_operator_1"] = "curvelet"
# constraint["TD_l1_sigma_1"]    = 0.5*norm(C*x,1)

# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
# (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

# (x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
# result=Vector{typeof(x[1])}(length(x1))
# for i=1:length(TD_OP)-1
#   copy!(result,x1)
#   @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
# end


end
