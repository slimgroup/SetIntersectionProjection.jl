export default_PARSDMM_options

function default_PARSDMM_options(options,TF)

  if     TF==Float64
    TI = Int64
  elseif TF == Float32
    TI = Int32
  end

  options.x_min_solver          = "CG_normal" #"CG_normal_plus_GMG","CG_normal_plus_AMG", "CG_normal_plus_ParSpMatVec","AMG"
  options.maxit                 = TI(200)
  options.evol_rel_tol          = TF(1e-4)
  options.feas_tol              = TF(5e-2)
  options.obj_tol               = TF(1e-3)
  options.rho_ini               = [TF(10.0)]
  options.rho_update_frequency  = TI(2)
  options.gamma_ini             = TF(1.0)
  options.adjust_rho            = true
  options.adjust_gamma          = true
  options.adjust_feasibility_rho= true
  options.Blas_active           = true
  options.linear_inv_prob_flag  = false
  options.FL                    = TF
  options.parallel              = false
  options.zero_ini_guess        = true

  return options
end
