
using DistributedArrays
using LinearAlgebra

# # add at least 3 worker processes
if nworkers() < 3
    n = max(3, min(8, Sys.CPU_CORES))
    addprocs(n; exeflags=`--check-bounds=yes`)
end
@assert nprocs() > 3
@assert nworkers() >= 3

@everywhere importall DistributedArrays
@everywhere importall JOLI
using Base.Test
@everywhere using SetIntersectionProjection

@everywhere mutable struct compgrid
  d :: Tuple
  n :: Tuple
end



@testset "SetIntersectionProjection" begin

include("test_projectors.jl")

#still need to port the stuff below to Julia 1.1

# include("test_TD_OPs.jl")
# include("test_prox_l2s!.jl")
# include("test_argmin_x.jl")
# include("test_update_y_l.jl")
#
# #parallel scripts
# include("test_update_y_l_parallel.jl")
# include("test_adapt_rho_gamma_parallel.jl")
#
# #linear algebra subroutines
# include("test_cg.jl")
# include("test_CDS_Mvp.jl")
# include("test_CDS_scaled_add.jl")
# include("test_Q_update.jl")
#
# #test full algorithms
# include("test_PARSDMM.jl")
# include("test_PARSDMM_parallel.jl")
# include("test_PARSDMM_multilevel.jl")

end
