#Illustrate how to set up certain constraint sets that are not found in other examples

using SetIntersectionProjection
using MAT
using LinearAlgebra
using Statistics
using PyPlot

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

#computational grid for the training images (all images assumed to be on the same grid here)
comp_grid  = compgrid((1f0,1f0,1f0),(size(m_tensor,1),size(m_tensor,2), size(m_tensor,3)))

######################################################################
######## (anisotropic) total-variation on the time-coordinate ########

#the video has coordinates x-y-time, so the time corresponds to the 3rd coordinate
#which is z in the SetIntersectionProjection system (x-y-z)

#initialize constraints
constraint = Vector{SetIntersectionProjection.set_definitions}()

#l1 (total variation) constraints (in one direction)

#find a reasonable value for the l1-ball
(TD_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",options.FL)
TV_z = norm(TD_OP*vec(m_tensor),1)

m_min     = 0.0
m_max     = 0.1*TV_z
set_type  = "l1"
TD_OP     = "D_z"
app_mode  = ("tensor","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

options=PARSDMM_options()
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

@time (x,log_PARSDMM) = PARSDMM(vec(m_tensor),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(vec(m_tensor),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

m_proj = reshape(x,comp_grid.n)

figure();
for i=1:comp_grid.n[3]
    subplot(2,1,1);imshow(m_proj[:,:,i],cmap="gray");title("projected");
    subplot(2,1,2);imshow(m_tensor[:,:,i],cmap="gray");title("original");
    pause(0.025)
end