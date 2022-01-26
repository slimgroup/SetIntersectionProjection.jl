@testset "multilevel_PARSDMM" begin

Random.seed!(123)

#test serial multilevel PARSDMM
    TF = Float32
    options = PARSDMM_options()
    default_PARSDMM_options(options,TF)
    options.parallel     = false
    options.evol_rel_tol = 100*eps(TF)
    options.maxit        = 5000

    n_levels          = 2
    coarsening_factor = 1.87

    x         = randn(TF,100,201)
    comp_grid = compgrid((1.0, 1.0),(100, 201))
    x         = vec(x)
    m         = deepcopy(x)

    constraint = Vector{SetIntersectionProjection.set_definitions}()

    #total variation
    (TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",options.FL)
    m_min     = 0.0
    m_max     = 0.5f0*norm(TD_OP*x,1)
    set_type  = "l1"
    TD_OP     = "TV"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
    
    #(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
    #(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)
    (TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
   
    (x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  
    result = deepcopy(x1)
    for i=1:length(TD_OP)-1
      copy!(result,x1)
      @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= 1.5*options.feas_tol
    end

#test parallel multilevel PARSDMM
        options=PARSDMM_options()
        default_PARSDMM_options(options,Float64)
        options.parallel = true

            (TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,constraint_level)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)

            (x2,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
            result=Vector{typeof(x[1])}(length(x2))
            for i=1:length(TD_OP_levels[1])-1
              copy!(result,x2)
              @test norm(P_sub_levels[1][i](TD_OP_levels[1][i]*result)-(TD_OP_levels[1][i]*result))/norm((TD_OP_levels[1][i]*result)) <= options.feas_tol
            end
@test isapprox(x1,x2,rtol=options.feas_tol)

#run serial PARSDMM and compare results
default_PARSDMM_options(options,Float64)
options.parallel    = false

(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL);
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

(x3,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
result=Vector{typeof(x[1])}(length(x3))
for i=1:length(TD_OP)-1
  copy!(result,x3)
  @test norm(P_sub[i](TD_OP[i]*result)-(TD_OP[i]*result))/norm((TD_OP[i]*result)) <= options.feas_tol
end

@test isapprox(x1,x3,rtol=options.feas_tol)

end
