export argmin_x

function argmin_x{TF<:Real}(
                  Q               ::Union{Array{TF,2},SparseMatrixCSC{TF,Int32},SparseMatrixCSC{TF,Int64}},
                  rhs             ::Vector{TF},
                  x               ::Vector{TF},
                  x_min_solver    ::String,
                  x_solve_tol_ref ::TF,
                  i               ::Integer,
                  log_PARSDMM,
                  Q_offsets=[],
                  Ax_out=zeros(TF,length(x)) ::Vector{TF}
                  )

#function argmin_x(Q::SparseMatrixCSC{Float64,Int64},rhs::Vector{Float64},tol::Float64,x::Vector{Float64},comp_grid,x_min_solver,i::Int64,log_PSDMM,rho::Vector{Float64},rho_old::Vector{Float64},A_g::Vector{SparseMatrixCSC{Float64,Int64}})
    #(x,iter,relres) = argmin_x(Q,rhs,1e-10,x,x_min_solver);
    flag    = 0
    reslres = 0.0
    iter    = 0

    # #determine what relative residual CG needs to reach
    # x_solve_tol_ref=TF(min.(0.0001,x_solve_tol_ref))
    # if i>1
    #   x_solve_tol_ref=min(0.1*log_PARSDMM.r_pri_total[i-1],x_solve_tol_ref) #makre sure CG is always more accurate than the current residual A*x-y
    #   if log_PARSDMM.cg_it[i-1]==1
    #     x_solve_tol_ref=0.5*x_solve_tol_ref #require more accuracy if previous PARSDMM iteration only used 1 CG iteration, as this is a waste of time
    #   end
    #   x_solve_tol_ref=TF(max(x_solve_tol_ref,10*eps(TF))) #dont' go below 10 X machine precision
    # end


  if x_min_solver == "CG_normal" # for package: KrylovMethods
    if typeof(Q)==Array{TF,2}
      Af(in) =  (fill!(Ax_out,0); CDS_MVp_MT(size(Q,1),size(Q,2),Q,Q_offsets,in,Ax_out); return Ax_out)
      if i<3
        x_solve_tol_ref = max(TF(0.1)*norm(Af(x)-rhs)/norm(rhs),TF(10)*eps(TF))
      else
        x_solve_tol_ref = min(max(TF(0.1)*norm(Af(x)-rhs)/norm(rhs),TF(10)*eps(TF)),x_solve_tol_ref)
      end
      #x_solve_tol_ref = min(TF(0.1)*norm(Af(x)-rhs)/norm(rhs),x_solve_tol_ref)
      (x,flag,relres,iter) = cg(Af,rhs,tol=x_solve_tol_ref,maxIter=1000,x=x,out=0);
    else
      (x,flag,relres,iter) = cg(Q,rhs,tol=x_solve_tol_ref,maxIter=1000,x=x,out=0);
    end

  elseif x_min_solver == "CG_normal_plus_Jacobi"
      Prec(x)= x./diag(Q);
      (x,flag,relres,iter) = cg(Q,rhs,tol=x_solve_tol_ref,maxIter=1000,M=Prec,x=x[:],out=0);
  # elseif x_min_solver == "CG_normal_plus_ParSpMatVec" # very slow somehow
  #     yt = Vector{Float64}(length(rhs))#zeros(size(Q,1));
  #     nthreads=4;
  #     Afun(x) = (yt[:]=0.0; ParSpMatVec.A_mul_B!( 1.0, Q, x, 0.0, yt, nthreads); return yt)
  #     (x,flag,relres,iter) = cg(Afun,rhs,tol=tol,maxIter=1000,x=x[:],out=0)
    end
    x_solve_tol_ref=TF(x_solve_tol_ref)
  return x,iter,relres,x_solve_tol_ref
end
