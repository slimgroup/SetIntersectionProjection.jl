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

"""
solve the x-minimization step in PARSDMM, i.e., x-minimization w.r.t. the augmented Lagrangian
"""

    #Initialize
    flag    = 0
    reslres = 0.0
    iter    = 0

  if x_min_solver == "CG_normal" # for package: KrylovMethods
    if typeof(Q)==Array{TF,2}
      #set up multi-threaded matrix-vector product in compressed diagonal storage format
      Af(in) =  (fill!(Ax_out,0); CDS_MVp_MT(size(Q,1),size(Q,2),Q,Q_offsets,in,Ax_out); return Ax_out)

      #determine what relative residual CG needs to reach
      if i<3 #i is the PARSDMM iteration counter
        x_solve_tol_ref = TF(max(0.1*norm(Af(x)-rhs)/norm(rhs),10*eps(TF))) #10% of current relative residual
      else
        x_solve_tol_ref = TF(min(max(0.1*norm(Af(x)-rhs)/norm(rhs),10*eps(TF)),x_solve_tol_ref)) #10% of current relative residual, but not larger than previous relative residual
      end

      (x,flag,relres,iter) = cg(Af,rhs,tol=x_solve_tol_ref,maxIter=1000,x=x,out=0);
    else
      #CG with native Julia sparse CSC format MPVs
      (x,flag,relres,iter) = cg(Q,rhs,tol=x_solve_tol_ref,maxIter=1000,x=x,out=0);
    end

    end

    x_solve_tol_ref=TF(x_solve_tol_ref)
  return x,iter,relres,x_solve_tol_ref
end
