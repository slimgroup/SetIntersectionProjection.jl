#Illustrate how to set up certain constraint sets that are not found in other examples

using SetIntersectionProjection
using MAT
using LinearAlgebra, SparseArrays
using Statistics
using PyPlot
using JOLI 

#define computational grid structure
mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

#select working precision
TF=Float32 #Float32

#load escalator video) (Mablab files for compatibility with matlab only solvers for comparison...)
if isfile("escalator_data.mat") == true
  file     = matopen("escalator_data.mat")
  #file = matopen(joinpath(dirname(pathof(SetIntersectionProjection)), "../examples/Data/escalator_data.mat"))
else
  println("downloading escalator video from http://cvxr.com/tfocs/demos/rpca/escalator_data.mat")
  run(`wget http://cvxr.com/tfocs/demos/rpca/escalator_data.mat`)
  file     = matopen("escalator_data.mat")
  #error("download escalator video from http://cvxr.com/tfocs/demos/rpca/escalator_data.mat")
end

mtrue    = read(file, "X")
n1       = convert(Integer,read(file, "m"))
n2       = convert(Integer,read(file, "n"))
m_mat    = convert(Array{TF,2},mtrue)
m_tensor = convert(Array{TF,3},reshape(mtrue,n1,n2,Integer(200)))

#computational grid for the video
comp_grid  = compgrid((1f0,1f0,1f0),(size(m_tensor,1),size(m_tensor,2), size(m_tensor,3)))

comp_grid_time_slice = compgrid((1f0,1f0),(size(m_tensor,1),size(m_tensor,2)))


######################################################################
######## (anisotropic) total-variation on the time-coordinate ########

#the video has coordinates x-y-time, so the time corresponds to the 3rd coordinate
#which is z in the SetIntersectionProjection system (x-y-z)

#initialize constraints
constraint = Vector{SetIntersectionProjection.set_definitions}()
options = PARSDMM_options()

#l1 (total variation) constraints (in one direction)

#find a reasonable value for the l1-ball
(TD_OP, AtA_diag, dense, TD_n) = get_TD_operator(comp_grid,"D_z",options.FL)
TV_z = norm(TD_OP*vec(m_tensor),1)

m_min     = 0.0
m_max     = 0.025*TV_z
set_type  = "l1"
TD_OP     = "D_z"
app_mode  = ("tensor","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

@time (x,log_PARSDMM) = PARSDMM(vec(m_tensor),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(vec(m_tensor),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

m_proj = reshape(x,comp_grid.n)

figure();
for i=1:comp_grid.n[3]
    subplot(2,1,1);imshow(m_proj[:,:,i],cmap="gray");title("projected");
    subplot(2,1,2);imshow(m_tensor[:,:,i],cmap="gray");title("original");
    pause(0.01)
end


###############################################################################
######## (anisotropic) total-variation on the time-derivative          ########
########                     using JOLI operators                      ########

options = PARSDMM_options()

#TV operator per time-slice
(TV, AtA_diag, dense, TD_n) = get_TD_operator(comp_grid_time_slice,"TV",options.FL)
#time derivative over the time-slices
(D, AtA_diag, dense, TD_n)  =  get_TD_operator(comp_grid,"D_z",options.FL)

CustomOP_explicit_sparse = kron(TV, SparseMatrixCSC{TF}(LinearAlgebra.I, comp_grid.n[3]-1,comp_grid.n[3]-1))* D

D  = joMatrix(D)
TV = joMatrix(TV)
CustomOP_JOLI = joKron(TV, joEye(comp_grid.n[3]-1,DDT=Float32,RDT=Float32))* D

##Solve using JOLI##

    #initialize constraints
    constraint = Vector{SetIntersectionProjection.set_definitions}()

    m_min     = 0.0
    m_max     = 0.1*norm(CustomOP_JOLI*vec(m_tensor),1)
    set_type  = "l1"
    TD_OP     = "identity"
    app_mode  = ("matrix","")
    custom_TD_OP = (CustomOP_JOLI,false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

    (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)

    #set properties of custom operator
    set_Prop.AtA_diag[1]  = false
    set_Prop.dense[1]     = false
    set_Prop.banded[1]    = false

    (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

    @time (x_joli,log_PARSDMM) = PARSDMM(vec(m_tensor),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
    @time (x_joli,log_PARSDMM) = PARSDMM(vec(m_tensor),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

##solve using explicit sparse array ##
    constraint = Vector{SetIntersectionProjection.set_definitions}()

    m_min     = 0.0
    m_max     = 0.025*norm(CustomOP_explicit_sparse*vec(m_tensor),1)
    set_type  = "l1"
    TD_OP     = "identity"
    app_mode  = ("matrix","")
    custom_TD_OP = (CustomOP_explicit_sparse,false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

    options=PARSDMM_options()
    (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)

    #set properties of custom operator
    set_Prop.AtA_diag[1]  = false
    set_Prop.dense[1]     = false
    set_Prop.banded[1]    = true

    (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

    @time (x_sp,log_PARSDMM) = PARSDMM(vec(m_tensor),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
    @time (x_sp,log_PARSDMM) = PARSDMM(vec(m_tensor),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

    m_proj = reshape(x_sp,comp_grid.n)

figure();
for i=1:comp_grid.n[3]
    subplot(2,1,1);imshow(m_proj[:,:,i],cmap="gray");title("projected");
    subplot(2,1,2);imshow(m_tensor[:,:,i],cmap="gray");title("original");
    pause(0.025)
end


# #every time-slice should have a small TV like this: || TV(slice_2 -slice_1) ||_1 <= sigma, || TV(slice_3 -slice_1) ||_1 <= sigma, ...

# #initialize constraints
# constraint = Vector{SetIntersectionProjection.set_definitions}()

# slice_ref = copy(m_tensor[:,:,1])

# m_shifted = similar(m_tensor)
# for i=1:comp_grid.n[3]
#     m_shifted[:,:,i] = m_tensor[:,:,i] - slice_ref
# end

# #find a reasonable value for the l1-ball
# (TD_OP, AtA_diag, dense, TD_n) = get_TD_operator(comp_grid,"D_z",options.FL)
# TV_z = norm(TD_OP*vec(m_shifted),1)

# m_min     = 0.0
# m_max     = 0.1*TV_z
# set_type  = "l1"
# TD_OP     = "TV"
# app_mode  = ("matrix","")
# custom_TD_OP = ([],false)
# push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

# options=PARSDMM_options()
# (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
# (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

# @time (x,log_PARSDMM) = PARSDMM(vec(m_shifted),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
# @time (x,log_PARSDMM) = PARSDMM(vec(m_shifted),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

# m_proj = reshape(x,comp_grid.n)
# for i=1:comp_grid.n[3]
#     m_proj[:,:,i] = m_proj[:,:,i] + slice_ref
# end

# figure();
# for i=1:comp_grid.n[3]
#     subplot(2,1,1);imshow(m_proj[:,:,i],cmap="gray");title("projected");
#     subplot(2,1,2);imshow(m_tensor[:,:,i],cmap="gray");title("original");
#     pause(0.025)
# end


###############################################################################
########################## custom dense JOLI matrix  ##########################

x = randn(Float32,10,20)
comp_grid  = compgrid((1f0,1f0),(10,20))

constraint = Vector{SetIntersectionProjection.set_definitions}()

options=PARSDMM_options()

#l1:
m_min     = 0.0f0
m_max     = 10.0f0
set_type  = "l1"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = (joMatrix(randn(options.FL, 133,prod(comp_grid.n))),false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)

#set properties of custom operator
set_Prop.AtA_diag[1]  = false
set_Prop.dense[1]     = true
set_Prop.banded[1]    = false

options.adjust_rho             = false
options.adjust_feasibility_rho = false

(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

@time (x_proj,log_PARSDMM) = PARSDMM(vec(x),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x_proj,log_PARSDMM) = PARSDMM(vec(x),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
