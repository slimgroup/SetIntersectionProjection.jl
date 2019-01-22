export adapt_rho_gamma_parallel

"""
Barzilai-Borwein scaling for Douglash-Rachford splitting on the dual problem related to standard ADMM.
Updates relaxation and augmented-Lagrangian penalty parameter.
To be used in parallel version of PARSDMM.
"""
function adapt_rho_gamma_parallel(
                                  gamma           ::Vector{TF},
                                  rho             ::Vector{TF},
                                  adjust_gamma    ::Bool,
                                  adjust_rho      ::Bool,
                                  y               ::Vector{Vector{TF}},
                                  y_old           ::Vector{Vector{TF}},
                                  s               ::Vector{Vector{TF}},
                                  s_0             ::Vector{Vector{TF}},
                                  l               ::Vector{Vector{TF}},
                                  l_hat_0         ::Vector{Vector{TF}},
                                  l_0             ::Vector{Vector{TF}},
                                  l_old           ::Vector{Vector{TF}},
                                  y_0             ::Vector{Vector{TF}},
                                  l_hat           ::Vector{Vector{TF}},
                                  d_l_hat         ::Vector{Vector{TF}},
                                  d_H_hat         ::Vector{Vector{TF}},
                                  d_l             ::Vector{Vector{TF}},
                                  d_G_hat         ::Vector{Vector{TF}}
                                  ) where {TF<:Real}

  const eps_correlation = TF(0.3) #hardcoded and suggested value by the paper based on numerical evidence


      if TF==Float64
          safeguard = 1e-10
      elseif TF==Float32
          safeguard = 1f-6
      end

      #standard loop-fusion
       @. l_hat[1]   = l_old[1] + rho[1]* ( -s[1] + y_old[1] )
      # @. d_l_hat[1] = l_hat[1] - l_hat_0[1]
      # @. d_H_hat[1] = s[1] - s_0[1]
      # @. d_l[1]     = l[1] - l_0[1]
      # @. d_G_hat[1] = -(y[1] - y_0[1])

      #wrap multi-theading inside a function
      a_is_b_min_c_MT!(d_l_hat[1],l_hat[1],l_hat_0[1])
      a_is_b_min_c_MT!(d_H_hat[1],s[1],s_0[1])
      a_is_b_min_c_MT!(d_l[1],l[1],l_0[1])
      a_is_b_min_c_MT!(d_G_hat[1],y_0[1],y[1])


      d_dHh_dlh   = dot(d_H_hat[1],d_l_hat[1])
      d_dGh_dl    = dot(d_G_hat[1],d_l[1])

      n_d_H_hat   = norm(d_H_hat[1])
      n_d_l_hat   = norm(d_l_hat[1])
      n_d_l       = norm(d_l[1])
      n_d_G_hat   = norm(d_G_hat[1])

      alpha_comp = false
      #print(alpha_comp)
      if (n_d_H_hat*n_d_l_hat) > safeguard && (n_d_H_hat.^2) > safeguard && d_dHh_dlh>safeguard
        alpha_correlation = d_dHh_dlh./( n_d_H_hat*n_d_l_hat  )
        if alpha_correlation > eps_correlation
          alpha_comp = true
          alpha_hat_MG  = d_dHh_dlh./(n_d_H_hat.^2);
          alpha_hat_SD  = (n_d_l_hat^2)./d_dHh_dlh;
          if (TF(2.0)*alpha_hat_MG) > alpha_hat_SD
            alpha_hat = alpha_hat_MG;
          else
            alpha_hat = alpha_hat_SD - alpha_hat_MG/TF(2.0);
          end
        end
      end

      beta_comp = false
      if (n_d_G_hat*n_d_l) > safeguard && (n_d_G_hat^2) > safeguard && d_dGh_dl > safeguard
        beta_correlation  = d_dGh_dl./( n_d_G_hat*n_d_l )
        if beta_correlation > eps_correlation
          beta_comp = true
          beta_hat_MG = d_dGh_dl./(n_d_G_hat^2);
          beta_hat_SD = (n_d_l^2)./d_dGh_dl;
          if (TF(2.0)*beta_hat_MG) > beta_hat_SD
            beta_hat = beta_hat_MG;
          else
            beta_hat = beta_hat_SD - beta_hat_MG/TF(2.0);
          end
        end
      end

      #update rho and or gamma
    if adjust_rho == true  && adjust_gamma == true
      if alpha_comp == true && beta_comp == true
        rho[1] = sqrt(alpha_hat*beta_hat);
        gamma[1]=TF(1.0)+(( TF(2.0)*sqrt(alpha_hat*beta_hat) )./( alpha_hat+beta_hat ));
      elseif alpha_comp == true && beta_comp == false
        rho[1] = alpha_hat;
        gamma[1]=TF(1.9);
      elseif  alpha_comp == false && beta_comp == true
        rho[1] = beta_hat;
        gamma[1]=TF(1.1);
      else
        #rho = rho; #do nothing
        gamma[1]=TF(1.5);
      end
    elseif adjust_rho == true  && adjust_gamma == false
        if alpha_comp == true && beta_comp == true
          rho[1] = sqrt(alpha_hat*beta_hat);
        elseif alpha_comp == true && beta_comp == false
          rho[1] = alpha_hat;
        elseif alpha_comp == false && beta_comp == true
          rho[1] = beta_hat;
        else
          #rho = rho; #do nothing
        end
      elseif adjust_rho == false  && adjust_gamma == true
        if alpha_comp == true && beta_comp == true
          gamma[1]=TF(1.0)+(( TF(2.0)*sqrt(alpha_hat*beta_hat) )./( alpha_hat+beta_hat ));
        elseif alpha_comp == true && beta_comp == false
          gamma[1]=TF(1.9);
        elseif  alpha_comp == false && beta_comp == true
          gamma[1]=TF(1.1);
        else
          gamma[1]=TF(1.5);
        end
      end #end compute new rho and gamma



# copy!(l_hat_0[1],l_hat[1])
# copy!(y_0[1],y[1])
# copy!(s_0[1],s[1])
# copy!(l_0[1],l[1])

#return rho,gamma,l_hat,d_l_hat,d_H_hat,d_l,d_G_hat
end #function adapt_rho_gamma
