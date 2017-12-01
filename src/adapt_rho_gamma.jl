export adapt_rho_gamma

function adapt_rho_gamma{TF<:Real}(
                                  i               ::Integer,
                                  gamma           ::Vector{TF},
                                  rho             ::Vector{TF},
                                  adjust_gamma    ::Bool,
                                  adjust_rho      ::Bool,
                                  adjust_rho_type ::String,
                                  y               ::Vector{Vector{TF}},
                                  y_old           ::Vector{Vector{TF}},
                                  s               ::Vector{Vector{TF}},
                                  s_0             ::Vector{Vector{TF}},
                                  l               ::Vector{Vector{TF}},
                                  l_hat_0         ::Vector{Vector{TF}},
                                  l_0             ::Vector{Vector{TF}},
                                  l_old           ::Vector{Vector{TF}},
                                  y_0             ::Vector{Vector{TF}},
                                  p               ::Integer,
                                  l_hat           ::Vector{Vector{TF}},
                                  d_l_hat         ::Vector{Vector{TF}},
                                  d_H_hat         ::Vector{Vector{TF}},
                                  d_l             ::Vector{Vector{TF}},
                                  d_G_hat         ::Vector{Vector{TF}}
                                  )

    # Barzilai-Borwein type for Douglash-Rachford splitting on the dual problem
    #hardcoded and suggested value by the paper based on numerical evidence

    if TF==Float64
        safeguard = 1e-10
    elseif TF==Float32
        safeguard = 1f-6
    end

if adjust_rho_type == "BB"
  const eps_correlation = TF(0.3);

  #Threads.@threads for ii = 1:p
  for ii = 1:p
      l_hat[ii]   .= l_old[ii] .+ rho[ii].* ( -s[ii] .+ y_old[ii] )
      d_l_hat[ii] .= l_hat[ii] .- l_hat_0[ii]
      d_H_hat[ii] .= s[ii] .- s_0[ii]


      d_dHh_dlh   = dot(d_H_hat[ii],d_l_hat[ii])
      n_d_H_hat   = norm(d_H_hat[ii])
      n_d_l_hat   = norm(d_l_hat[ii])
      d_l[ii]    .= l[ii] .- l_0[ii]
      n_d_l       = norm(d_l[ii])
      d_G_hat[ii].= -(y[ii].-y_0[ii])
      n_d_G_hat   = norm(d_G_hat[ii])
      d_dGh_dl    = dot(d_G_hat[ii],d_l[ii])

      alpha_reliable = false;
      if (n_d_H_hat*n_d_l_hat) > safeguard && (n_d_H_hat.^2) > safeguard && d_dHh_dlh>safeguard
        alpha_reliable = true
        alpha_correlation = d_dHh_dlh./( n_d_H_hat*n_d_l_hat  );
      end

      beta_reliable = false;
      if (n_d_G_hat*n_d_l) > safeguard && (n_d_G_hat^2) > safeguard && d_dGh_dl > safeguard
        beta_reliable = true
        beta_correlation  = d_dGh_dl./( n_d_G_hat*n_d_l );
      end

      alpha_comp = false
      if (alpha_reliable==true) && (alpha_correlation > eps_correlation)
        alpha_comp = true
        alpha_hat_MG  = d_dHh_dlh./(n_d_H_hat.^2);
        alpha_hat_SD  = (n_d_l_hat^2)./d_dHh_dlh;
        if (TF(2.0)*alpha_hat_MG) > alpha_hat_SD
          alpha_hat = alpha_hat_MG;
        else
          alpha_hat = alpha_hat_SD - alpha_hat_MG/TF(2.0);
        end
      end

      beta_comp = false
      if (beta_reliable==1) && (beta_correlation > eps_correlation)
        beta_comp = true
        beta_hat_MG = d_dGh_dl./(n_d_G_hat^2);
        beta_hat_SD = (n_d_l^2)./d_dGh_dl;
        if (TF(2.0)*beta_hat_MG) > beta_hat_SD
          beta_hat = beta_hat_MG;
        else
          beta_hat = beta_hat_SD - beta_hat_MG/TF(2.0);
        end
      end

      #update rho and or gamma
      if adjust_rho == true  && adjust_gamma == false
        if alpha_comp == true && beta_comp == true
          rho[ii] = sqrt(alpha_hat*beta_hat);
        elseif alpha_comp == true && beta_comp == false
          rho[ii] = alpha_hat;
        elseif alpha_comp == false && beta_comp == true
          rho[ii] = beta_hat;
        else
          #rho = rho; #do nothing
        end
      elseif adjust_rho == true  && adjust_gamma == true
        if alpha_comp == true && beta_comp == true
          rho[ii] = sqrt(alpha_hat*beta_hat);
          gamma[ii]=TF(1.0)+(( TF(2.0)*sqrt(alpha_hat*beta_hat) )./( alpha_hat+beta_hat ));
        elseif alpha_comp == true && beta_comp == false
          rho[ii] = alpha_hat;
          gamma[ii]=TF(1.9);
        elseif  alpha_comp == false && beta_comp == true
          rho[ii] = beta_hat;
          gamma[ii]=TF(1.1);
        else
          #rho = rho; #do nothing
          gamma[ii]=TF(1.5);
        end
      elseif adjust_rho == false  && adjust_gamma == true
        if alpha_comp == true && beta_comp == true
          gamma[ii]=TF(1.0)+(( TF(2.0)*sqrt(alpha_hat*beta_hat) )./( alpha_hat+beta_hat ));
        elseif alpha_comp == true && beta_comp == false
          gamma[ii]=TF(1.9);
        elseif  alpha_comp == false && beta_comp == true
          gamma[ii]=TF(1.1);
        else
          gamma[ii]=TF(1.5);
        end
      end #end compute new rho and gamma
    end #end for loop

end #adjust rho

return rho,gamma,l_hat,d_l_hat,d_H_hat,d_l ,d_G_hat
end #function adapt_rho_gamma
