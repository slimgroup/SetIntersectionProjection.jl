export Dykstra_prox_parallel2
function Dykstra_prox_parallel2(
                                        x::Vector{TF},
                                        P::Vector{Any},
                                        P_sub,
                                        TD_OP,
                                        closed_form,
                                        maxit_dyk,
                                        dyk_feas_tol,
                                        obj_dyk_tol
                                        ) where {TF<:Real}

#this script computes the projection of x0 onto the intersection of and
#arbitrary number of closed and convex sets. May still work in case
#nonconvex sets are used.
# Solves: min_x 0.5||x-x0||_2^2 s.t. x in intersection of m convex sets.
#
# This algorithm allows for parallel evaluation of the prox part
# (projections). This version does it in serial anyway, so no Matlab
# Parallel computing toolbox is required.
#
# input:
#       x                   -   vector to be projection onto the intersection
#       P                   -   contains projectors. P{1},P{2}...P{m} as generated by setup_constraint.m in folder :
#       options_PARSDMM.
#           options_PARSDMM.tol     -   tolerance based on residual as stopping condition
#           options_PARSDMM.maxIt   -   maximum number of Parallel proximal Dykstra-like iterations
#           options_PARSDMM.minIt   -   minimum number of Parallel proximal Dykstra-like iterations
#           options_PARSDMM.log_vec -   if (=1), saves all vectors at every iteration for analysis purposes.
#           options_PARSDMM.feas_tol-   feasibility tolerance for warning message, not for stopping condition
#           options_PARSDMM.evol_rel_tol- tolerance on relative evolution between iterations: exit if norm(x-x_old)/norm(x_old) becomes too small
#
# output:
#       x           -   result
#       res         -   relative residual
#       prog_rel    -   relative evolution (movement) per iteration

# Author: Bas Peters


# Initialize
N       = length(x)
m = length(P) #number of constraints
ptp   = Vector{TF}(undef,N)
x_old = Vector{TF}(undef,N)
copyto!(ptp, x)
copyto!(x_old,x)
z=Vector{Vector{TF}}(undef,m)
p=Vector{Vector{TF}}(undef,m)

obj=zeros(TF,maxit_dyk)
rel_feasibility_err=zeros(TF,maxit_dyk+1,m)

svd_P=zeros(Int64,maxit_dyk)
bounds_P=zeros(Int64,maxit_dyk)
ARADMM_it=zeros(Int64,maxit_dyk)
cg_it=zeros(Int64,maxit_dyk)

for i=1:m
    z[i]=zeros(TF,N)
    copyto!(z[i],x)
    p[i]=zeros(TF,N)
end
omega   = 1.0 ./ m; #weights are hardcored here

#initial feasibility (transform-domain, relative)
#check distance to feasibility
for i=1:m
  rel_feasibility_err[1,i]=norm(P_sub[i](TD_OP[i]*x)-TD_OP[i]*x)/norm(TD_OP[i]*x)
end


## Main loop
for n=1:maxit_dyk

    #evaluate prox
    temp1=0
    temp2=0
    for i=1:m #make this loop parallel for actual parallel proximal evaluation
        if closed_form[i]==false
          (p[i],log_PARSDMM) = P[i](z[i])
          if m==2; temp1= length(log_PARSDMM.obj); end
          if m==2; temp2= sum(log_PARSDMM.cg_it); end
          if m==2; svd_P[n] = length(log_PARSDMM.obj); end
        else
          copyto!(p[i],P[i](z[i]))
        end
  end
  ARADMM_it[n] = temp1
  cg_it[n] = temp2

    #averaging step
    if n>1; copyto!(x_old,x); end;
    fill!(x,TF(0.0))
    for i=1:m
        x .= x .+ omega.*p[i];
    end

    #log objective
    obj[n].=0.5*norm(ptp-x,2).^2;

    #updating
    for i=1:m
        z[i] .= x .+ z[i] .- p[i];
    end

    #check distance to feasibility
    for i=1:m
      rel_feasibility_err[n+1,i]=norm(P_sub[i](TD_OP[i]*x)-TD_OP[i]*x)/norm(TD_OP[i]*x)
    end

    #stop if objective value does not change and x is sufficiently feasible for all sets
    if n>1 && (maximum(rel_feasibility_err[n+1,:])<dyk_feas_tol) && ( ( (obj[n]-obj[n-1])./obj[n] ) < obj_dyk_tol )
        println("stationary objective and reached feasibility, exiting Dykstra (iteration ",n,")")
        obj=obj[1:n]
        cg_it=cg_it[1:n]
        svd_P=svd_P[1:n]
        ARADMM_it=ARADMM_it[1:n]
        rel_feasibility_err=rel_feasibility_err[1:n+1,:]
        return x,obj,rel_feasibility_err,cg_it,ARADMM_it,svd_P
    end
end

#obj=obj[1:n]
#rel_feasibility_err=rel_feasibility_err[1:n,:]

return x,obj,rel_feasibility_err,cg_it,ARADMM_it,svd_P
end
