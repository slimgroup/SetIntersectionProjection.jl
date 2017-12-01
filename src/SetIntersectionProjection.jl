module SetIntersectionProjection


using Parameters
using Interpolations
using DistributedArrays
using JOLI
export log_type_PARSDMM, transform_domain_properties, PARSDMM_options

#main scripts
include("PARSDMM.jl")
include("PARSDMM_multi_level.jl")
include("PARSDMM_precompute_distribute.jl")

#algorithms and solver scripts
include("argmin_x.jl");
include("adapt_rho_gamma.jl");
include("update_y_l.jl");
include("stop_PARSDMM.jl");
include("PARSDMM_initialize.jl")
include("Q_update!.jl")

#linear algebra functions
include("cg.jl");
include("mat2CDS.jl")
include("CDS_MVp.jl")
include("CDS_MVp_MT.jl")
include("CDS_MVp_MT_subfunc.jl")
include("CDS_scaled_add!.jl")

#scripts for parallelism
include("update_y_l_parallel.jl")
include("adapt_rho_gamma_parallel.jl")
include("compute_relative_feasibility.jl")

#multi-threaded scripts
include("a_is_b_min_c_MT!.jl")

#multi level related scripts
include("setup_multi_level_PARSDMM.jl")
include("constraint2coarse.jl")
include("interpolate_y_l.jl")

#scripts to set up constraints
include("default_PARSDMM_options.jl")
include("convert_options!.jl")
include("get_discrete_Grad.jl");
include("get_TD_operator.jl");
include("get_bound_constraints.jl");
include("setup_constraints.jl");

#projectors
include("projectors/project_bounds!.jl")
include("projectors/project_cardinality!.jl");
include("projectors/project_rank!.jl");
include("projectors/project_nuclear!.jl");
include("projectors/project_l1_Duchi!.jl");
include("projectors/project_l2!.jl");
include("projectors/project_subspace!.jl");

include("prox_l2s!.jl")

#scripts for linear inverse problems
#include("ICLIP_inpainting.jl")

#scripts that are required to run examples
include("constraint_learning_by_observation.jl")

#define types
mutable struct log_type_PARSDMM
      set_feasibility   :: Array{Real,2}
      r_dual            :: Array{Real,2}
      r_pri             :: Array{Real,2}
      r_dual_total      :: Vector{Real}
      r_pri_total       :: Vector{Real}
      obj               :: Vector{Real}
      evol_x            :: Vector{Real}
      rho               :: Array{Real,2}
      gamma             :: Array{Real,2}
      cg_it             :: Vector{Integer}
      cg_relres         :: Vector{Real}
      T_cg              :: Real
      T_stop            :: Real
      T_ini             :: Real
      T_rhs             :: Real
      T_adjust_rho_gamma:: Real
      T_y_l_upd         :: Real
      T_Q_upd           :: Real
end

@with_kw mutable struct PARSDMM_options
  x_min_solver          :: String  = "CG_normal" #"CG_normal_plus_GMG","CG_normal_plus_AMG", "CG_normal_plus_ParSpMatVec","AMG"
  maxit                 :: Integer   = 200
  evol_rel_tol          :: Real = 1e-4
  feas_tol              :: Real = 5e-2
  obj_tol               :: Real = 1e-3
  rho_ini               :: Real = 10.0
  rho_update_frequency  :: Integer   = 2
  gamma_ini             :: Real = 1.0
  adjust_rho            :: Bool    = true
  adjust_gamma          :: Bool    = false
  adjust_feasibility_rho:: Bool    = true
  adjust_rho_type       :: String  = "BB"
  Blas_active           :: Bool    = true
  linear_inv_prob_flag  :: Bool    = false
  FL                    :: DataType = Float32
  parallel              :: Bool    = false
  zero_ini_guess        :: Bool    = true
end

type transform_domain_properties
           ncvx       ::Vector{Bool}
           AtA_diag   ::Vector{Bool}
           dense      ::Vector{Bool}
           TD_n       ::Vector{Tuple}
           tag        ::Vector{Tuple{String,String}}
           banded     ::Vector{Bool}
           AtA_offsets::Union{Vector{Vector{Int32}},Vector{Vector{Int64}}}
end

end # module
