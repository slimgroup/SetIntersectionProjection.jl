@testset "PARSDMM" begin
#test PARSDMM
Random.seed!(123)

options=PARSDMM_options()
options.FL                    = Float64
options.parallel              = false
options.zero_ini_guess        = true

x=randn(100,201)

comp_grid = compgrid((1.0, 1.0),(100, 201))
x         = vec(x)
m2        = deepcopy(x)
x_ini2    = deepcopy(x)

#check if input returns if the input satisfies the constraints
  #bound constraints
  constraint = Vector{SetIntersectionProjection.set_definitions}()
  
  m_min     = minimum(vec(x))
  m_max     = maximum(vec(x))
  set_type  = "bounds"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)
  
  m=deepcopy(x)

  #default settings
  (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  @test x==m

#add a few convex constraints. The projected model should satisfy these constraints
#(a necessary but not sufficient test for the projection onto the intersection)
 
  constraint = Vector{SetIntersectionProjection.set_definitions}()

  #bound constraints
  m_min     = 0.5*minimum(vec(x))
  m_max     = 0.5*maximum(vec(x))
  set_type  = "bounds"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  #transform domain bounds
  (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",options.FL)
  m_min     = 0.5*minimum(TD_OP*x)
  m_max     = 0.5*maximum(TD_OP*x)
  set_type  = "bounds"
  TD_OP     = "D_z"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  #total variation
  (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",options.FL)
  m_min     = 0.0
  m_max     = 0.5*norm(TD_OP*x,1)
  set_type  = "l1"
  TD_OP     = "TV"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

  m=deepcopy(x)

  #(mostly) default settings
  options              = PARSDMM_options()
  options.FL           = Float64
  options.evol_rel_tol = 10*eps()
  options.maxit        = 5000

  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

  result = deepcopy(x)
  for i=1:length(TD_OP)-1
    copy!(result,x)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  #accurate setting
  options              = PARSDMM_options()
  options.FL           = Float64
  options.obj_tol      = 1e-12
  options.feas_tol     = 1e-12
  options.evol_rel_tol = 10*eps()
  options.maxit        = 5000

  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  result = deepcopy(x)
  
  for i=1:length(TD_OP)-1
    copy!(result,x)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  #accurate without BLAS
  options               = PARSDMM_options()
  options.FL            = Float64
  options.Blas_active   = false
  options.obj_tol       = 1e-12
  options.feas_tol      = 1e-12
  options.evol_rel_tol  = 10*eps()
  options.maxit         = 5000
  
  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  result = deepcopy(x)
  for i=1:length(TD_OP)-1
    copy!(result,x)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  #accurate with rho adjustment, without gamma adjustment
  options               = PARSDMM_options()
  options.FL            = Float64
  options.adjust_gamma  = false
  options.obj_tol       = 1e-12
  options.feas_tol      = 1e-12
  options.evol_rel_tol  = 10*eps()
  options.maxit         = 5000

  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  result = deepcopy(x)
 for i=1:length(TD_OP)-1
    copy!(result,x)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  #accurate without rho adjustment, with gamma
  options               = PARSDMM_options()
  options.FL            = Float64
  options.adjust_gamma  = true
  options.adjust_rho    = false
  options.obj_tol       = 1e-6
  options.feas_tol      = 1e-6
  options.evol_rel_tol  = 10*eps()
  options.maxit         = 10000

  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  result = deepcopy(x)
 for i=1:length(TD_OP)-1
    copy!(result,x)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  # without rho adjustment, without gamma
  options               = PARSDMM_options()
  options.FL            = Float64
  options.adjust_gamma  = false
  options.adjust_rho    = false
  options.obj_tol       = 1e-6
  options.feas_tol      = 1e-6
  options.evol_rel_tol  = 10*eps()
  options.maxit         = 25000

  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  result = deepcopy(x)
  for i=1:length(TD_OP)-1
    copy!(result,x)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  #accurate with rho adjustment, with gamma and without rho based on feasibility
  options                        = PARSDMM_options()
  options.FL                     = Float64
  options.adjust_feasibility_rho = false
  options.adjust_gamma           = true
  options.adjust_rho             = true
  options.obj_tol                = 1e-12
  options.feas_tol               = 1e-12
  options.evol_rel_tol           = 10*eps()
  options.maxit                  = 5000

  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  result = deepcopy(x)
  for i=1:length(TD_OP)-1
    copy!(result,x)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end


# some closed form solutions:

  #create a vector that has a certain nuclear norm as reference solution
  x         = randn(100,201)
  comp_grid = compgrid((1.0, 1.0),(100, 201))
  x         = vec(x)
  
  c_l_solution = deepcopy(x)
  tau          = 0.54321
  c_l_solution = reshape(c_l_solution,comp_grid.n)

  project_nuclear!(c_l_solution,tau,[])
  c_l_solution = vec(c_l_solution)

  constraint = Vector{SetIntersectionProjection.set_definitions}()
 
  #add nuclear norm constraint
    m_min     = 0.0
    m_max     = tau
    set_type  = "nuclear"
    TD_OP     = "identity"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

  #solve
  m = deepcopy(x)

  options                        = PARSDMM_options()
  options.FL                     = Float64
  options.adjust_feasibility_rho = true
  options.adjust_gamma           = true
  options.adjust_rho             = true
  options.obj_tol                = 10*eps()
  options.feas_tol               = 10*eps()
  options.evol_rel_tol           = 10*eps()
  options.maxit                  = 2500
  options.Blas_active            = false
  (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

  result = deepcopy(x)
  for i=1:length(TD_OP)-1
    copy!(result,x)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= 2.0*options.feas_tol
  end
  result = deepcopy(x)
  @test norm(result-c_l_solution)/norm(c_l_solution) <= 1e-9

#test if the algorithm gives same output if we use many explicit BLAS calls
x         = randn(100,201)
comp_grid = compgrid((1.0, 1.0),(100, 201))
x         = vec(x)

constraint = Vector{SetIntersectionProjection.set_definitions}()
 
#add nuclear norm constraint
  m_min     = 0.0
  m_max     = 1.123
  set_type  = "nuclear"
  TD_OP     = "identity"
  app_mode  = ("matrix","")
  custom_TD_OP = ([],false)
  push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#add a few convex constraints
  #bound constraints
    m_min     = 1.0*minimum(x)
    m_max     = 0.50*maximum(x)
    set_type  = "bounds"
    TD_OP     = "identity"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  #transform domain bounds
    (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",options.FL)
    m_min     = 0.9*minimum(TD_OP*vec(x))
    m_max     = 0.67*maximum(TD_OP*vec(x))
    set_type  = "bounds"
    TD_OP     = "D_z"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

  #total variation
    (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",options.FL)
    m_min     = 0.0
    m_max     =  0.2*norm(TD_OP*vec(x),1)
    set_type  = "l1"
    TD_OP     = "TV"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
  
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

  #solve
  m = deepcopy(x)

  options                        = PARSDMM_options()
  options.FL                     = Float64
  options.adjust_feasibility_rho = true
  options.adjust_gamma           = true
  options.adjust_rho             = true
  options.obj_tol                = 1e-6
  options.feas_tol               = 1e-6
  options.evol_rel_tol           = 1e-6
  options.maxit                  = 2500
 
  options.Blas_active = false
  (x_noblas,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

  m = deepcopy(x)
  options.Blas_active=true
  (x_blas,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

 #put test here
 @test isapprox(x_blas,x_noblas,rtol=1e-12)

end
