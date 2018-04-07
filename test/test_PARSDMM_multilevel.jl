@testset "multilevel_PARSDMM" begin

# if nworkers()==1
#   addprocs(3)
# end

#test serial multilevel PARSDMM
    options=PARSDMM_options()
    default_PARSDMM_options(options,Float64)
    options.parallel = false
    options.evol_rel_tol = 10*eps()
    options.maxit=5000

    n_levels=2
    coarsening_factor=1.87

    x=randn(100,201)

    comp_grid=compgrid((1.0, 1.0),(100, 201))
    x         = vec(x)
    m         = deepcopy(x)

    constraint=Dict()

    #total variation
    (TV_OP, AtA_diag, dense, TD_n, banded)=get_TD_operator(comp_grid,"TV",options.FL)
    constraint["use_TD_l1_1"]      = true
    constraint["TD_l1_operator_1"] = "TV"
    constraint["TD_l1_sigma_1"]    = 0.4*norm(TV_OP*x,1)

    (m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,constraint_level)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)

    (x1,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
    result=Vector{typeof(x[1])}(length(x1))
    for i=1:length(TD_OP_levels[1])-1
      copy!(result,x1)
      @test norm(P_sub_levels[1][i](TD_OP_levels[1][i]*result)-(TD_OP_levels[1][i]*result))/norm((TD_OP_levels[1][i]*result)) <= options.feas_tol
    end

#test parallel multilevel PARSDMM
        options=PARSDMM_options()
        default_PARSDMM_options(options,Float64)
        options.parallel = true

            (m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,constraint_level)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)

            (x2,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
            result=Vector{typeof(x[1])}(length(x2))
            for i=1:length(TD_OP_levels[1])-1
              copy!(result,x2)
              @test norm(P_sub_levels[1][i](TD_OP_levels[1][i]*result)-(TD_OP_levels[1][i]*result))/norm((TD_OP_levels[1][i]*result)) <= options.feas_tol
            end
@test isapprox(x1,x2,rtol=options.feas_tol)

#run serial PARSDMM and compare results
default_PARSDMM_options(options,Float64)
options.parallel    = false

(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

(x3,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
result=Vector{typeof(x[1])}(length(x3))
for i=1:length(TD_OP)-1
  copy!(result,x3)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
end

@test isapprox(x1,x3,rtol=options.feas_tol)

end
