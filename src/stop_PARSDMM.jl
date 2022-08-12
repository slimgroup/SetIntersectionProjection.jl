export stop_PARSDMM

"""
implement stopping conditions for PARSDMM.jl
if activated, we set the flag stop==true and pass it back to PARSDMM.jl
"""
function stop_PARSDMM(
                    log_PARSDMM,
                    i                        ::Integer,
                    evol_rel_tol             ::TF,
                    feas_tol                 ::TF,
                    obj_tol                  ::TF,
                    adjust_rho               ::Bool,
                    adjust_gamma             ::Bool,
                    adjust_feasibility_rho   ::Bool,
                    ind_ref                  ::Integer,
                    counter                  ::Integer
                    ) where {TF<:Real}

    stop = false;

    #stop if objective value does not change and x is sufficiently feasible for all sets
    if i>6 && maximum(log_PARSDMM.set_feasibility[counter-1,:])<feas_tol && maximum(abs.( (log_PARSDMM.obj[i-5:i]-log_PARSDMM.obj[i-1-5:i-1])./log_PARSDMM.obj[i-1-5:i-1] )) < obj_tol
        constr_log("stationary objective and reached feasibility, exiting PARSDMM (iteration ",i,")")
        stop=true;
    end

    #stop if x doesn't change significantly anyjore
    if i>5 && maximum(log_PARSDMM.evol_x[i-5:i])<evol_rel_tol
      constr_log("relative evolution to small, exiting PARSDMM (iteration ",i,")")
      stop=true;
    end

    # fix rho to ensure regular ADMM convergence if primal residual does not decrease over a 20 iteration window
    if i>20 && adjust_rho==true && log_PARSDMM.r_pri_total[i]>maximum(log_PARSDMM.r_pri_total[(i-1):-1:max((i-50),1)])
      constr_log("no primal residual reduction, fixing PARSDMM rho & gamma (iteration ",i,")")
      adjust_rho = false;
      adjust_feasibility_rho = false;
      adjust_gamma = false;
      if nprocs()>1
        [ @spawnat pid adjust_gamma for pid in workers() ]
        [ @spawnat pid adjust_rho for pid in workers() ]
        [ @spawnat pid adjust_feasibility_rho for pid in workers() ]
      end
      ind_ref = i;
    end

    #if rho is fixed and still no decrease in primal residual is observed over a window, we give up
    if adjust_rho==false && i>(ind_ref+25) && log_PARSDMM.r_pri_total[i]>maximum(log_PARSDMM.r_pri_total[(i-1):-1:max(ind_ref,max((i-50),1))])
      constr_log("no primal residual reduction, exiting PARSDMM (iteration ",i,")")
      stop = true;
    end
    return stop,adjust_rho,adjust_gamma,adjust_feasibility_rho,ind_ref
end #stop_psdmm
