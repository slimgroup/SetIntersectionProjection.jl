@testset "PARSDMM" begin
#test PARSDMM

options=PARSDMM_options()
options.FL                    = Float64
options.parallel              = false
options.zero_ini_guess        = true

x=randn(100,201)

comp_grid = compgrid((1.0, 1.0),(100, 201))
x         = vec(x)
m2        = deepcopy(x)
x_ini2    = deepcopy(x)
constraint= Dict()

#check if input returns if the input satisfies the constraints
  #bound constraints
  constraint["use_bounds"]=true
  constraint["min"]=minimum(vec(x))
  constraint["max"]=maximum(vec(x))

  (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

  m=deepcopy(x)

  #default settings
  (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  @test x==m

#add a few convex constraints. The projected model should satisfy these constraints
  #bound constraints
  constraint["use_bounds"]=true
  constraint["min"]=0.5*minimum(vec(x))
  constraint["max"]=0.5*maximum(vec(x))

  #transform domain bounds
  (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",options.FL)
  constraint["use_TD_bounds_1"]=true
  constraint["TDB_operator_1"]="D_z";
  constraint["TD_LB_1"]=0.5*minimum(TD_OP*x)
  constraint["TD_UB_1"]=0.5*maximum(TD_OP*x)

  #nuclear norm

  #total variation
  (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",options.FL)
  constraint["use_TD_l1_1"]     = true
  constraint["TD_l1_operator_1"] = "TV"
  constraint["TD_l1_sigma_1"]     = 0.5*norm(TD_OP*x,1)

  (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
  (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

  m=deepcopy(x)

  #default settings
  options=PARSDMM_options()
  options.FL                    = Float64
  options.evol_rel_tol = 10*eps()
  options.maxit=5000
  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  result=Vector{typeof(x[1])}(length(x))
  for i=1:length(TD_OP)-1
  copy!(result,x)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  #accurate
  options=PARSDMM_options()
  options.FL                    = Float64
  options.obj_tol  = 1e-12
  options.feas_tol = 1e-12
  options.evol_rel_tol = 10*eps()
  options.maxit=5000
  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  result=Vector{typeof(x[1])}(length(x))
  for i=1:length(TD_OP)-1
  copy!(result,x)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  #accurate without BLAS
  options=PARSDMM_options()
  options.FL                    = Float64
  options.Blas_active = false
  options.obj_tol  = 1e-12
  options.feas_tol = 1e-12
  options.evol_rel_tol = 10*eps()
  options.maxit=5000
  (x2,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  result=Vector{typeof(x[1])}(length(x2))
  for i=1:length(TD_OP)-1
  copy!(result,x2)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  #accurate without gamma adjustment
  options=PARSDMM_options()
  options.FL                    = Float64
  options.adjust_gamma = false
  options.obj_tol  = 1e-12
  options.feas_tol = 1e-12
  options.evol_rel_tol = 10*eps()
  options.maxit=5000
  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  result=Vector{typeof(x[1])}(length(x))
  for i=1:length(TD_OP)-1
  copy!(result,x)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  #accurate without rho adjustment, with gamma
  options=PARSDMM_options()
  options.FL                    = Float64
  options.adjust_gamma = true
  options.adjust_rho = false
  options.obj_tol  = 1e-6
  options.feas_tol = 1e-6
  options.evol_rel_tol = 10*eps()
  options.maxit=10000
  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  result=Vector{typeof(x[1])}(length(x))
  for i=1:length(TD_OP)-1
  copy!(result,x)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  # without rho adjustment, without gamma
  options=PARSDMM_options()
  options.FL                    = Float64
  options.adjust_gamma = false
  options.adjust_rho = false
  options.obj_tol  = 1e-6
  options.feas_tol = 1e-6
  options.evol_rel_tol = 10*eps()
  options.maxit=25000
  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  result=Vector{typeof(x[1])}(length(x))
  for i=1:length(TD_OP)-1
  copy!(result,x)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end

  #accurate with rho adjustment, with gamma and with rho based on feasibility
  options=PARSDMM_options()
  options.FL                    = Float64
  options.adjust_feasibility_rho = true
  options.adjust_gamma = true
  options.adjust_rho = true
  options.obj_tol  = 1e-12
  options.feas_tol = 1e-12
  options.evol_rel_tol = 10*eps()
  options.maxit=5000
  (x,log_PARSDMM) = PARSDMM(m2,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  result=Vector{typeof(x[1])}(length(x))
  for i=1:length(TD_OP)-1
  copy!(result,x)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
  end


# some closed form solutions:
  x=randn(100,201)
  comp_grid=compgrid((1.0, 1.0),(100, 201))
  x=vec(x)
  c_l_solution = deepcopy(x)
  tau=0.54321
  c_l_solution=reshape(c_l_solution,comp_grid.n)
  project_nuclear!(c_l_solution,tau)
    c_l_solution=vec(c_l_solution)

  constraint=Dict()

  #add nuclear norm constraint
    constraint["use_TD_nuclear_1"]=true
    constraint["TD_nuclear_norm_1"]=tau
    constraint["TD_nuclear_operator_1"]="identity"

  #add a few convex constraints which the model already satisfies
    #bound constraints
    constraint["use_bounds"]=false
    constraint["min"]=1.0*minimum(vec(c_l_solution))
    constraint["max"]=1.456*maximum(vec(c_l_solution))

    #transform domain bounds
    (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",options.FL)
    constraint["use_TD_bounds_1"]=false
    constraint["TDB_operator_1"]="D_z";
    constraint["TD_LB_1"]=1.0*minimum(TD_OP*c_l_solution)
    constraint["TD_UB_1"]=1.1*maximum(TD_OP*c_l_solution)

    #total variation
    (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",options.FL)
    constraint["use_TD_l1_1"]     = false
    constraint["TD_l1_operator_1"] = "TV"
    constraint["TD_l1_sigma_1"]     = 1.2*norm(TD_OP*c_l_solution,1)

    (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
    (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

    #solve
    m=deepcopy(x)
    options=PARSDMM_options()
    options.FL                    = Float64
    options.adjust_feasibility_rho = true
    options.adjust_gamma = true
    options.adjust_rho = true
    options.obj_tol  = 1e-12
    options.feas_tol = 1e-12
    options.evol_rel_tol = 10*eps()
    options.maxit=2500
      options.Blas_active=false
    (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
    result=Vector{typeof(x[1])}(length(x))
    for i=1:length(TD_OP)-1
    copy!(result,x)
    @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= 2.0*options.feas_tol
    end
    @test norm(x-c_l_solution)/norm(c_l_solution) <= 1e-9

#test if the algorithm gives same output if we use many explicit BLAS calls
x=randn(100,201)
comp_grid=compgrid((1.0, 1.0),(100, 201))
x=vec(x)

constraint=Dict()

#add nuclear norm constraint
constraint["use_TD_nuclear_1"]=true
constraint["TD_nuclear_norm_1"]=1.123
constraint["TD_nuclear_operator_1"]="identity"

#add a few convex constraints which the model already satisfies
  #bound constraints
  constraint["use_bounds"]=true
  constraint["min"]=1.0*minimum(x)
  constraint["max"]=0.50*maximum(x)

  #transform domain bounds
  (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",options.FL)
  constraint["use_TD_bounds_1"]=true
  constraint["TDB_operator_1"]="D_z";
  constraint["TD_LB_1"]=0.9*minimum(TD_OP*vec(x))
  constraint["TD_UB_1"]=0.67*maximum(TD_OP*vec(x))

  #total variation
  (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",options.FL)
  constraint["use_TD_l1_1"]     = true
  constraint["TD_l1_operator_1"] = "TV"
  constraint["TD_l1_sigma_1"]     = 0.2*norm(TD_OP*vec(x),1)

  (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
  (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

  #solve
  m=deepcopy(x)
  options=PARSDMM_options()
  options.FL                    = Float64
  options.adjust_feasibility_rho = false
  options.adjust_gamma = true
  options.adjust_rho = true
  options.obj_tol  = 1e-6
  options.feas_tol = 1e-6
  options.evol_rel_tol = 10*eps()
  options.maxit=2500

  options.Blas_active=false
  (x_noblas,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);

  options.Blas_active=true
  (x_blas,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);

 #put test here
 @test isapprox(x_blas,x_noblas,rtol=1e-12)

end
