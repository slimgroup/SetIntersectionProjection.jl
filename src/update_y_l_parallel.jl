export update_y_l_parallel

"""
  update l and compute y . This is a subfunction for PARSDMM.jl
"""
function update_y_l_parallel(
  x                     ::Vector{TF},
  i                     ::Integer,
  Blas_active           ::Bool,
  y                     ::Vector{Vector{TF}},
  y_old                 ::Vector{Vector{TF}},
  l                     ::Vector{Vector{TF}},
  l_old                 ::Vector{Vector{TF}},
  rho                   ::Vector{TF},
  gamma                 ::Vector{TF},
  prox                  ::Vector{Any},
  TD_OP                 ::Vector{<:Union{SparseMatrixCSC{TF,TI},joAbstractLinearOperator{TF,TF}}},
  P_sub,
  x_hat                 ::Vector{Vector{TF}},
  r_pri                 ::Vector{Vector{TF}},
  s                     ::Vector{Vector{TF}},
  set_feas              ::Vector{TF},
  feasibility_only=false::Bool
  ) where {TF<:Real,TI<:Integer}

  rho1 = TF(1.0) ./ rho[1];

  copy!(y_old[1],y[1]);
  copy!(l_old[1],l[1]);

  mul!(s[1],TD_OP[1],x) #s[1] = TD_OP[1]*x;
  if Blas_active
    if gamma[1]==1 #without relaxation
      copyto!(y[1],s[1])
      #y[1]       = prox[1]( BLAS.axpy!(-rho1[1],l[1],y[1]) )
      BLAS.axpy!(-rho1[1],l[1],y[1])
      y[1] = prox[1](y[1])
      @. r_pri[1] = -s[1]+y[1];
      BLAS.axpy!(rho[1],r_pri[1],l[1]);
    else #relaxed iterations
      @. x_hat[1] = gamma[1]*s[1]
      BLAS.axpy!(TF(1.0)-gamma[1],y[1],x_hat[1]);
      #y[1]       = copy(x_hat[1]);
      copyto!(y[1],x_hat[1])
      #y[1]       = prox[1]( BLAS.axpy!(-rho1[1],l[1],y[1]))
      BLAS.axpy!(-rho1[1],l[1],y[1])
      y[1] = prox[1](y[1])
       @. r_pri[1] = -s[1]+y[1];
      BLAS.axpy!(rho[1],y[1]-x_hat[1],l[1]);
    end
  else
    if gamma[1]==1 #without relaxation
      #y[1]       = prox[1]( s[1]-l[1]*rho1[1] );
       @. y[1]       = s[1]-l[1]*rho1[1]
       y[1]          = prox[1](y[1]);
       @. r_pri[1]   = -s[1]+y[1];
       @. l[1]       = l[1]+rho[1]*r_pri[1];
    else #relaxed iterations
       @. x_hat[1]   = gamma[1]*s[1] + ( TF(1.0)-gamma[1] )*y[1]
      #y[1]          = prox[1]( x_hat[1]-l[1]*rho1[1] );
       @. y[1]       = x_hat[1]-l[1]*rho1[1]
       y[1]          = prox[1]( y[1] )
       @. r_pri[1]   = -s[1]+y[1]
       @. l[1]       = l[1]+rho[1]*( -x_hat[1]+y[1] )
    end
  end #end blas

  #log_PARSDMM.r_pri[i,1]  = norm(r_pri[1])#./(norm(y[1])+(100*eps(TF))); #add 1-14, because y may be 0 for certain initializations of x,y,l
  #log r_pri outside this function for parallel use (temporarily)
  # if Blas_active
  #   log_PARSDMM.r_dual[i,1] = norm( BLAS.scal!(lx,rho[1],TD_OP[1]'*(y[1]-y_old[1]),1 ))
  # else
  #  log_PARSDMM.r_dual[i,1] = norm( rho[1]*(TD_OP[1]'*(y[1]-y_old[1])) )
  # end

  #log feasibility
  if (mod(i,10) == 0 && length(P_sub)==1 ) && ((feasibility_only == false && myid()<nprocs()) || feasibility_only == true)

    copy!(x_hat[1],s[1]) #use the already allocated x_hat as a temp placeholder
    P_sub[1](x_hat[1])   #P_sub mutates the input in-place: P(A*x), s=A*x, x_hat = s. The next iteration x_hat will be overwritten anyway before use
    set_feas[1] = norm( x_hat[1] .- s[1]) ./ (norm(s[1])+(100*eps(TF))); 
    
    #equivalent to, but with less allocations
    #log_PARSDMM.set_feasibility[counter,ii] = norm( P_sub[ii](deepcopy(s[ii])) .- s[ii]) ./ (norm(s[ii])+(100*eps(TF))); #use x_hat[ii] as output, because it is already allocated and not used afterwards. The next iteration it will be overwritten anyway before use

  end

#return y,l,r_pri,s,log_PARSDMM,counter,y_old,l_old,x_hat,set_feas_vec

end #end of function
