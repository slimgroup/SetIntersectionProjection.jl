export compute_relative_feasibility

"""
Compute transform-domain relative feasibility w.r.t. a set
r_feas =  ||P(A*x)-A*x||/||A*x||
Helper function that is intended for the parallel version only, where just a single
operator is sent to a worker where this script runs
"""
# function compute_relative_feasibility(m::Vector{TF}, feasibility_initial::Vector{TF},
#                                       TD_OP::Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joAbstractLinearOperator{TF,TF}}},
#                                        P_sub) where {TF<:Real,TI<:Integer}
function compute_relative_feasibility(m::Vector{TF},feasibility_initial::TF,
    TD_OP::Union{SparseMatrixCSC{TF,TI},JOLI.joAbstractLinearOperator{TF,TF}},
     P_sub) where {TF<:Real,TI<:Integer}

    #note that P_sub mutates the input in-place
    s_temp = TD_OP*m
    norm(P_sub(deepcopy(s_temp)) .- s_temp)
     # feasibility_initial[1]=norm(P_sub[1](pm) .- pm) ./ (norm(pm)+(100*eps(TF)))
    feasibility_initial = norm(P_sub(deepcopy(s_temp)) .- s_temp) ./ (norm(s_temp)+(100*eps(TF)))
    s_temp = []
    #gc()# check if we need this in julia >0.7
    return feasibility_initial
end
