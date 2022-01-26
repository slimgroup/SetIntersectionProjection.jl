export argmin_x

"""
solve the x-minimization step in PARSDMM, i.e., x-minimization w.r.t. the augmented Lagrangian
"""
function argmin_x(
                  Q               ::Union{Array{TF,2},SparseMatrixCSC{TF,Int64},joAbstractLinearOperator{TF, TF}},
                  rhs             ::Vector{TF},
                  x               ::Vector{TF},
                  x_solve_tol_ref ::TF,
                  i               ::Integer,
                  log_PARSDMM,
                  Q_offsets=[],
                  Ax_out=zeros(TF,length(x)) ::Vector{TF}
                  ) where {TF<:Real}

  #Initialize
  flag    = 0
  reslres = 0.0
  iter    = 0

  if typeof(Q)==Array{TF,2} #set up multi-threaded matrix-vector product in compressed diagonal storage format
    Af1(in) =  (fill!(Ax_out,TF(0)); CDS_MVp_MT(size(Q,1),size(Q,2),Q,Q_offsets,in,Ax_out); return Ax_out)

    #determine what relative residual CG needs to reach
    if i<3 #i is the PARSDMM iteration counter
      x_solve_tol_ref = TF(max(0.1*norm(Af1(x)-rhs)/norm(rhs),10*eps(TF))) #10% of current relative residual
    else
      x_solve_tol_ref = TF(min(max(0.1*norm(Af1(x)-rhs)/norm(rhs),10*eps(TF)),x_solve_tol_ref)) #10% of current relative residual, but not larger than previous relative residual
    end

    (x,flag,relres,iter) = cg(Af1,rhs,tol=x_solve_tol_ref,maxIter=1000,x=x,out=0);

  elseif typeof(Q)==SparseMatrixCSC{TF,Int64} #CG with native Julia sparse CSC format MPVs
    
    #determine what relative residual CG needs to reach
    if i<3 #i is the PARSDMM iteration counter
      x_solve_tol_ref = TF(max(0.1*norm(Q*x-rhs)/norm(rhs),10*eps(TF))) #10% of current relative residual
    else
      x_solve_tol_ref = TF(min(max(0.1*norm(Q*x-rhs)/norm(rhs),10*eps(TF)),x_solve_tol_ref)) #10% of current relative residual, but not larger than previous relative residual
    end

    (x,flag,relres,iter) = cg(Q,rhs,tol=x_solve_tol_ref,maxIter=1000,x=x,out=0);

  else #typeof(Q) <: joAbstractLinearOperator{TF, TF}#CG with JOLI operators
    Af2(in) =  return Q*in
    #determine what relative residual CG needs to reach
    if i<3 #i is the PARSDMM iteration counter
      x_solve_tol_ref = TF(max(0.1*norm(Q*x-rhs)/norm(rhs),10*eps(TF))) #10% of current relative residual
    else
      x_solve_tol_ref = TF(min(max(0.1*norm(Q*x-rhs)/norm(rhs),10*eps(TF)),x_solve_tol_ref)) #10% of current relative residual, but not larger than previous relative residual
    end


    (x,flag,relres,iter) = cg(Af2,rhs,tol=x_solve_tol_ref,maxIter=1000,x=x,out=0);

  end

  x_solve_tol_ref = TF(x_solve_tol_ref)

  return x,iter,relres,x_solve_tol_ref
end
