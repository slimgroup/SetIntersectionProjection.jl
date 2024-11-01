export cg


function cg(A::Union{SparseMatrixCSC{TF,Int},Matrix{TF}},b::Vector{TF}; kwargs...) where {TF<:Real}
	x = zeros(TF,size(A,2)) # pre-allocate
	return cg(v -> mul!(x, transpose(A), v, TF(1.0), TF(0.0)),b; kwargs...) # multiply with transpose of A for efficiency
end

cg(A,b::Vector;kwargs...)= cg(x -> A*x,b;kwargs...)

"""
This is a slightly modified version from the Julia Package KrylovMethods

x,flag,err,iter,resvec = cg(A,b,tol=1e-2,maxIter=100,M=1,x=[],out=0)


(Preconditioned) Conjugate Gradient applied to the linear system A*x = b, where A is assumed
to be symmetric positive semi definite.

Input:

  A       - matrix or function computing A*x
  b       - right hand side vector
  tol     - error tolerance
  maxIter - maximum number of iterations
  M       - preconditioner, a function that computes M\\x
  x       - starting guess
  out     - flag for output (-1: no output, 0: only errors, 1: final status, 2: residual norm at each iteration)

Output:

  x       - approximate solution
  flag    - exit flag ( 0 : desired tolerance achieved,
  					 -1 : maxIter reached without converging
  					 -2 : Matrix A is not positive definite
  					 -3 : cg stalled, i.e. two consecutive residuals have same norm
  					 -9 : right hand side was zero)
  err     - norm of relative residual, i.e., norm(A*x-b)/norm(b)
  iter    - number of iterations
  resvec  - norm of relative residual at each iteration
"""
# function cg(A::Function,b::Vector; tol::Real=1e-2,maxIter::Integer=100,M::Function=identity,x::Vector=[],out::Integer=0,
# 	storeInterm::Bool=false)
function cg(A::Function,b::Vector{TF}; tol::Real=1e-2,maxIter::Integer=100,M::Function=identity,x::Vector=[],out::Integer=0) where {TF <: Real}
	n = length(b)

	if norm(b)==0; return zeros(eltype(b),n), -9, 0f0, 0, [0f0]; end
	if isempty(x)
		x = zeros(TF,n)#x = zeros(eltype(b),n)
		r = copy(b)
	else
		r = b .- A(x)::Vector{TF}
	end
	z = M(r)::Vector{TF}
	p = copy(z)::Vector{TF}

    # if storeInterm
    #     X = zeros(n,maxIter)	# allocate space for intermediates
    # end

	nr0  = norm(b)


	if out==2
		constr_log("=== cg ===")
		constr_log(@sprintf("%4s\t%7s","iter","relres"))
	end

	resvec = zeros(TF,maxIter)
	iter   = 1 # makes iter available outside the loop
	flag   = -1

	if norm(r)/nr0<=tol
		flag = 0;
		return x,flag,resvec[iter],iter,resvec[1:iter]
	end

	#TF    = typeof(x[1])
 	alpha = TF(0)
	beta  = TF(0)
	lastIter = 0
	for iter=1:maxIter
		lastIter = iter

		Ap 	  = A(p)::Vector{TF}
		gamma = dot(r,z)
		#gamma = BLAS.dot(n, r,1,z,1)#
		alpha = gamma / dot(p,Ap)
		#alpha = gamma / BLAS.dot(n, p,1,Ap,1)#dot(p,Ap)

		if alpha==Inf || alpha<0
			flag = -2; break
		end

		BLAS.axpy!(n,alpha,p,1,x,1) # x = alpha*p+x
		#if storeInterm; X[:,iter] = x; end
		BLAS.axpy!(n,-alpha,Ap,1,r,1) # r -= alpha*Ap

		#resvec[iter] = BLAS.nrm2(n, r, 1) / nr0#
		resvec[iter]  = norm(r)/nr0
		if out==2
			constr_log(iter,resvec[iter])
		end
		if resvec[iter] <= tol
			flag = 0; break
		end

		z    = M(r)::Vector{TF}
		#beta = BLAS.dot(n, z,1,r,1) / gamma
		beta  = dot(z,r) / gamma
		# the following two lines are equivalent to p = z + beta*p
		#p = BLAS.scal!(n,beta,p,1)::Vector{TF}
		#p = BLAS.axpy!(n,TF(1.0),z,1,p,1)::Vector{TF}
		p  = BLAS.axpby!(TF(1.0), z, beta, p)::Vector{TF} # Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y
	end

	if out>=0
		if flag==-1
			constr_log("cg iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",
																								maxIter,resvec[lastIter],tol)
		elseif flag==-2
			constr_log("Matrix A in cg has to be positive definite.")
		elseif flag==0 && out>=1
			constr_log("cg achieved desired tolerance at iteration %d. Residual norm is %1.2e.",lastIter,resvec[lastIter])
		end
	end
    return x,flag,resvec[lastIter],lastIter,resvec[1:lastIter]
end

#specifically for CDS/DIA storage matrices
# function cg(R::Matrix{TF},offset::Vector{TI},b::Vector{TF},tol::Real,maxIter::Integer,x::Vector{TF},out::Integer) where {TF <: Real, TI <: Integer}
# 	n  	   = length(b)
# 	ndiags = size(R,2)
# 	r  	   = zeros(TF,n)

# 	if norm(b)==0; return zeros(eltype(b),n),-9,0.0,0,[0.0]; end

# 	CDS_MVp_MT(n,ndiags,R,offset,x,r)
# 	r .= -r .+ b
# 		#r = b .- A(x)::Vector{TF}

# 	z = copy(r)

# 	p = copy(r)::Vector{TF}

# 	nr0  = norm(b)

# 	resvec = zeros(TF,maxIter)
# 	iter   = 1 # makes iter available outside the loop
# 	flag   = -1

# 	if norm(r)/nr0<=tol
# 		flag = 0;
# 		return x,flag,resvec[iter],iter,resvec[1:iter]
# 	end

#  	alpha  	 = TF(0)
# 	beta     = TF(0)
# 	lastIter = 0
# 	Ap  	 = zeros(TF,n)
# 	for iter=1:maxIter
# 		lastIter = iter

# 		#Ap 	  = A(p)::Vector{TF}
# 		fill!(Ap,TF(0))
# 		CDS_MVp_MT(n,ndiags,R,offset,p,Ap)
# 		gamma = dot(r,z)
# 		#gamma = BLAS.dot(n, r,1,z,1)#
# 		alpha = gamma / dot(p,Ap)
# 		#alpha = gamma / BLAS.dot(n, p,1,Ap,1)#dot(p,Ap)

# 		if alpha==Inf || alpha<0
# 			flag = -2; break
# 		end

# 		BLAS.axpy!(n,alpha,p,1,x,1) # x = alpha*p+x
# 		BLAS.axpy!(n,-alpha,Ap,1,r,1) # r -= alpha*Ap

# 		#resvec[iter] = BLAS.nrm2(n, r, 1) / nr0#
# 		resvec[iter]  = norm(r)/nr0
# 		if resvec[iter] <= tol
# 			flag = 0; break
# 		end

# 		#z    = M(r)::Vector{TF}
# 		z = copy(r)
# 		#beta = BLAS.dot(n, z,1,r,1) / gamma
# 		beta  = dot(z,r) / gamma
# 		# the following two lines are equivalent to p = z + beta*p
# 		#p = BLAS.scal!(n,beta,p,1)::Vector{TF}
# 		#p = BLAS.axpy!(n,TF(1.0),z,1,p,1)::Vector{TF}
# 		p  = BLAS.axpby!(TF(1.0), z, beta, p)::Vector{TF} # Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y
# 	end
#
# 	if out>=0
# 		if flag==-1
# 			constr_log("cg iterated maxIter (=%d) times but reached only residual norm %1.2e instead of tol=%1.2e.",
# 																								maxIter,resvec[lastIter],tol)
# 		elseif flag==-2
# 			constr_log("Matrix A in cg has to be positive definite.")
# 		elseif flag==0 && out>=1
# 			constr_log("cg achieved desired tolerance at iteration %d. Residual norm is %1.2e.",lastIter,resvec[lastIter])
# 		end
# 	end

#     return x,flag,resvec[lastIter],lastIter,resvec[1:lastIter]

# end
