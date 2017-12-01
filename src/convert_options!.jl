export convert_options!

function convert_options!(options,TF,TI)

options.maxit                 = TI(options.maxit)
options.evol_rel_tol          = TF(options.evol_rel_tol)
options.feas_tol              = TF(options.feas_tol)
options.obj_tol               = TF(options.obj_tol )
options.rho_ini               = TF(options.rho_ini)
options.rho_update_frequency  = TI(options.rho_update_frequency)
options.gamma_ini             = TF(options.gamma_ini)
end
