@testset "parallel_PARSDMM" begin

# if nworkers()==1
#   addprocs(3)
# end

#test parallel PARSDMM (test is for 2 workers)
    options=PARSDMM_options()
    default_PARSDMM_options(options,Float64)
    options.parallel = true
    options.evol_rel_tol = 10*eps()
    options.maxit=5000

    x=randn(100,201)

    comp_grid=compgrid((1.0, 1.0),(100, 201))
    x         = vec(x)
    m         = deepcopy(x)

    constraint=Dict()

    #total variation
    (TV_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",options.FL)
    constraint["use_TD_l1_1"]      = true
    constraint["TD_l1_operator_1"] = "TV"
    constraint["TD_l1_sigma_1"]    = 0.5*norm(TV_OP*x,1)

    (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
    (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(m,TD_OP,TD_Prop,comp_grid,options)

    (x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
    result=Vector{typeof(x[1])}(length(x1))
    for i=1:length(TD_OP)-1
      copy!(result,x1)
      @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
    end

#test parallel PARSDMM without many explicit BLAS calls (test is for 2 workers)
    default_PARSDMM_options(options,Float64)
    options.parallel    = true
    options.Blas_active = false

    (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
    (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(m,TD_OP,TD_Prop,comp_grid,options)

    (x2,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
    result=Vector{typeof(x[1])}(length(x2))
    for i=1:length(TD_OP)-1
      copy!(result,x2)
      @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
    end

#test if with and ithout expicit Blas calls results are the same
@test isapprox(x1,x2,rtol=1e-12)

#run serial PARSDMM and compare results
    default_PARSDMM_options(options,Float64)
    options.parallel    = false

    (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
    (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(m,TD_OP,TD_Prop,comp_grid,options)

    (x3,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
    result=Vector{typeof(x[1])}(length(x3))
    for i=1:length(TD_OP)-1
      copy!(result,x3)
      @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
    end

@test isapprox(x1,x3,rtol=1e-12)


#test parallel PARDMM with a JOLI operator in a projector, but not in TD_OP
default_PARSDMM_options(options,Float64)
options.parallel    = false
options.evol_rel_tol = 10*eps()
constraint=Dict()

#DFT l1 constraints
(DFT, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"DFT",options.FL)
constraint["use_TD_l1_1"]      = true
constraint["TD_l1_operator_1"] = "DFT"
constraint["TD_l1_sigma_1"]    = 0.5*norm(DFT*x,1)

(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(m,TD_OP,TD_Prop,comp_grid,options)

(x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
result=Vector{typeof(x[1])}(length(x1))
for i=1:length(TD_OP)-1
  copy!(result,x1)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
end

#test parallel PARDMM with a JOLI operator in TD_OP
default_PARSDMM_options(options,Float64)
options.parallel    = false

constraint=Dict()

#DFT l1 constraints
(C, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"curvelet",options.FL)
constraint["use_TD_l1_1"]      = true
constraint["TD_l1_operator_1"] = "curvelet"
constraint["TD_l1_sigma_1"]    = 0.5*norm(C*x,1)

(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(m,TD_OP,TD_Prop,comp_grid,options)

(x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
result=Vector{typeof(x[1])}(length(x1))
for i=1:length(TD_OP)-1
  copy!(result,x1)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
end


end
