export default_PARSDMM_options

function default_PARSDMM_options(options,TF)
"""
Returns a set of default options for the PARSDMM solver
"""

  if     TF == Float64
    TI = Int64
  elseif TF == Float32
    TI = Int32
  end

  options.x_min_solver          = "CG_normal" #what algorithm to use for the x-minimization (CG appied to normal equations)
  options.maxit                 = TI(200)     #max number of PARSDMM iterations
  options.evol_rel_tol          = TF(1e-3)    #stop PARSDMM if ||x^k - X^{k-1}||_2 / || x^k || < options.evol_rel_tol AND options.feas_tol is reached
  options.feas_tol              = TF(5e-2)    #stop PARSDMM if the transform-domain relative feasibility error is < options.feas_tol AND options.evol_rel_tol is reached
  options.obj_tol               = TF(1e-3)    #optional stopping criterion for change in distance from point that we want to project
  options.rho_ini               = [TF(10.0)]  #initial values for the augmented-Lagrangian penalty parameters. One value in array or one value per constraint set in array
  options.rho_update_frequency  = TI(2)       #update augmented-Lagrangian penalty parameters and relaxation parameters every X number of PARSDMM iterations
  options.gamma_ini             = TF(1.0)     #initial value for all relaxation parameters (scalar)
  options.adjust_rho            = true        #adapt augmented-Lagrangian penalty parameters or not
  options.adjust_gamma          = true        #adapt relaxation parameters in PARSDMM
  options.adjust_feasibility_rho= true        #adapt augmented-Lagrangian penalty parameters based on constraint set feasibility errors (can be used in combination with options.adjust_rho)
  options.Blas_active           = true        #use direct BLAS calls, otherwise the code will use Julia loop-fusion where possible
  options.feasibility_only      = false       #drop distance term and solve a feasibility problem
  options.FL                    = TF          #type of Float: Float32 or Float64
  options.parallel              = false       #comput proximal mappings, multiplier updates, rho and gamma updates in parallel
  options.zero_ini_guess        = true        #zero initial guess for primal, auxilliary, and multipliers
  Minkowski                     = false       #the intersection of sets includes a Minkowski set
  return options
end
