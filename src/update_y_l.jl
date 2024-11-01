export update_y_l

"""
  update l and compute y . This is a subfunction for PARSDMM.jl
"""
function update_y_l(
  x                     ::Vector{TF},
  p                     ::Integer,
  i                     ::Integer,
  Blas_active           ::Bool,
  y                     ::Vector{Vector{TF}},
  y_old                 ::Vector{Vector{TF}},
  l                     ::Vector{Vector{TF}},
  l_old                 ::Vector{Vector{TF}},
  rho                   ::Vector{TF},
  gamma                 ::Vector{TF},
  prox                  ::Vector{Any},
  TD_OP                 ::Vector{<:Union{SparseMatrixCSC{TF,TI}, joAbstractLinearOperator{TF,TF}}},
  log_PARSDMM,
  P_sub,
  counter               ::Integer,
  x_hat                 ::Vector{Vector{TF}},
  r_pri                 ::Vector{Vector{TF}},
  s                     ::Vector{Vector{TF}},
  feasibility_only=false::Bool
  ) where {TF<:Real,TI<:Integer}


#x_hat   = Vector{Vector{Float64}}(p);
#r_pri   = Vector{Vector{Float64}}(p);
#s       = Vector{Vector{Float64}}(p)
lx   = length(x);
rho1 = Vector{TF}(undef,p)
rho1 = TF(1.0) ./ rho;

for ii=1:p
  #y_old[ii]=deepcopy(y[ii])
  #l_old[ii]=deepcopy(l[ii])
  copy!(y_old[ii],y[ii]);
  copy!(l_old[ii],l[ii]);

  
  mul!(s[ii],TD_OP[ii],x) #s[ii] = TD_OP[ii]*x;

  if Blas_active  #&& i>5
    if gamma[ii]==1 #without relaxation
      copyto!(y[ii],s[ii])
      #y[ii]       = prox[ii]( BLAS.axpy!(-rho1[ii],l[ii],y[ii]) )
      BLAS.axpy!(-rho1[ii],l[ii],y[ii])
      y[ii] = prox[ii](y[ii])
      r_pri[ii]   .= -s[ii] .+ y[ii];
      BLAS.axpy!(rho[ii],r_pri[ii],l[ii]);
    else #relaxed iterations
      x_hat[ii]   .= gamma[ii].*s[ii]
      BLAS.axpy!(TF(1.0) .- gamma[ii],y[ii],x_hat[ii]);
      #y[ii]       = copy(x_hat[ii]);
      copyto!(y[ii],x_hat[ii])
      #y[ii]       = prox[ii]( BLAS.axpy!(-rho1[ii],l[ii],y[ii]))
      BLAS.axpy!(-rho1[ii],l[ii],y[ii])
      y[ii]      = prox[ii](y[ii])
      r_pri[ii] .= -s[ii] .+ y[ii];
      BLAS.axpy!(rho[ii],y[ii] .- x_hat[ii],l[ii]);
    end
  else #without explicit BLAS calls
    if gamma[ii]==1 #without relaxation
      #y[ii]       = prox[ii]( s[ii].-l[ii].*rho1[ii] );
      y[ii]       .= s[ii] .- l[ii].*rho1[ii]
      y[ii]        = prox[ii](y[ii]);
      r_pri[ii]   .= -s[ii] .+ y[ii];
      l[ii]       .= l[ii] .+ rho[ii] .* r_pri[ii];
    else #relaxed iterations
      x_hat[ii]   .= gamma[ii].*s[ii] .+ ( TF(1.0) .- gamma[ii] ) .* y[ii];
      #y[ii]       = prox[ii]( x_hat[ii].-l[ii].*rho1[ii] );
      y[ii]       .= x_hat[ii] .- l[ii] .* rho1[ii]
      y[ii]        = prox[ii]( y[ii] );
      r_pri[ii]   .= -s[ii] .+ y[ii];
      l[ii]       .= l[ii] .+ rho[ii] .* ( -x_hat[ii] .+ y[ii] );
    end
  end #end blas

  log_PARSDMM.r_pri[i,ii]  = norm(r_pri[ii])#./(norm(y[ii])+(100*eps(TF))); #add 1-14, because y may be 0 for certain initializations of x,y,l
  copy!(x_hat[ii],y[ii] .- y_old[ii])
  #log_PARSDMM.r_dual[i,ii] = rho[ii] .* norm( TD_OP[ii]'*(y[ii] .- y_old[ii]) )
  log_PARSDMM.r_dual[i,ii] = rho[ii] .* norm( TD_OP[ii]'*x_hat[ii] )

  #log feasibility using the Euclidean distance-to-set function
  #log every 10 it, or whatever number is suitable. Don't compute feasibility for entry 
  # p in case of the projection problem, because then it corresponds to the l2-prox for
  # the euclidean distance from the inital point.
  if mod(i,10) == 0 && ((feasibility_only == false && ii<p) || feasibility_only == true)

      copy!(x_hat[ii],s[ii]) #use the already allocated x_hat as a temp placeholder
      P_sub[ii](x_hat[ii])   #P_sub mutates the input in-place: P(A*x), s=A*x, x_hat = s. The next iteration x_hat will be overwritten anyway before use
      log_PARSDMM.set_feasibility[counter,ii] = norm( x_hat[ii] .- s[ii]) ./ (norm(s[ii])+(100*eps(TF))); 
      
      #equivalent to, but with less allocations
      #log_PARSDMM.set_feasibility[counter,ii] = norm( P_sub[ii](deepcopy(s[ii])) .- s[ii]) ./ (norm(s[ii])+(100*eps(TF))); #use x_hat[ii] as output, because it is already allocated and not used afterwards. The next iteration it will be overwritten anyway before use

    end

end #end y and l update loop

if mod(i,10)==0
  counter += 1;
end

return y,l,r_pri,s,log_PARSDMM,counter,y_old,l_old

end #end of function
