#This script illustates various constraints (one at a time), applied to a 2D model

using SparseArrays, LinearAlgebra, Random
using JOLI
using SetIntersectionProjection
using MAT
using PyPlot

mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

#PARSDMM options:
options    = PARSDMM_options()
options.FL = Float32
#options=default_PARSDMM_options(options,options.FL)
options.adjust_gamma           = true
options.adjust_rho             = true
options.adjust_feasibility_rho = true
options.Blas_active            = true
options.maxit                  = 500

set_zero_subnormals(true)
BLAS.set_num_threads(2)

#select working precision
if options.FL==Float64
  TF = Float64
  TI = Int64
elseif options.FL==Float32
  TF = Float32
  TI = Int32
end

#load image to project
file = matopen(joinpath(dirname(pathof(SetIntersectionProjection)), "../examples/Data/compass_velocity.mat"));
m    = read(file, "Data");close(file);
m    = m[1:341,200:600];
m    = permutedims(m,[2,1]);

#set up computational grid (25 and 6 m are the original distances between grid points)
comp_grid = compgrid((TF(25.0), TF(6.0)),(size(m,1), size(m,2)))
m         = convert(Vector{TF},vec(m));

#define axis limits and colorbar limits
xmax = comp_grid.d[1]*comp_grid.n[1]
zmax = comp_grid.d[2]*comp_grid.n[2]
vmi  = 1500
vma  = 4500

##############################################################################
## Constraint 1: Lateral smoothness via bound constraints on the derivative ##

# define constraint
constraint = Vector{SetIntersectionProjection.set_definitions}()

#slope constraints (lateral)
m_min     = -0.2
m_max     = 0.2
set_type  = "bounds"
TD_OP     = "D_x"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP));

options.parallel       = false
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL);
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options);

@time (x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
#@time (x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

#plot
figure();imshow(permutedims(reshape(m,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
savefig("original_model.png",bbox_inches="tight")
figure();imshow(permutedims(reshape(x1,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds on lateral derivative")
savefig("projected_model.png",bbox_inches="tight")

##########################################################################
## Constraint 2: Lateral smoothness via l2 constraint on the derivative ##

# define constraint
constraint = Vector{SetIntersectionProjection.set_definitions}()

#l2 constraint on gradient (lateral)
m_min     = 0.0
m_max     = 20.0
set_type  = "l2"
TD_OP     = "D_x"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP));

options.parallel       = false
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL);
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options);

@time (x2,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
#@time (x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

#plot
figure();imshow(permutedims(reshape(m,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
savefig("original_model.png",bbox_inches="tight")
figure();imshow(permutedims(reshape(x2,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (l2 constraint on lateral derivative")
savefig("projected_model.png",bbox_inches="tight")


##########################################################################
## Constraint 4: Histogram constraints ##

using TestImages
using SetIntersectionProjection
using PyPlot

#goal observe histogram of mandril and make the cameraman have the same histogram
mandril   = testimage("mandril_gray")
mandril   = convert(Array{Float32,2},mandril)
cameraman = testimage("cameraman")
cameraman = convert(Array{Float32,2},cameraman)

#observe sorted values of mandril
obs = sort(vec(mandril))

output = project_histogram_relaxed!(vec(deepcopy(cameraman)),obs,obs)
output = reshape(output,size(cameraman))

figure();
subplot(2,3,1);imshow(mandril);title("madril image")
subplot(2,3,2);imshow(cameraman);title("cameraman image")
subplot(2,3,3);imshow(output);title("Cameraman image with histogram of mandril")
subplot(2,3,4);hist(vec(mandril));title("madril histogram")
subplot(2,3,5);hist(vec(cameraman));title("cameraman histogram")
subplot(2,3,6);hist(vec(output));title("adjusted cameraman histogram")

#now set this up again, this time using the higher level tools in SetIntersectionProjection
comp_grid = compgrid((TF(1.0), TF(1.0)),(size(cameraman,1), size(cameraman,2)))

#construct permutation matrix
#P1 y <= P2 x <= P1 y
#<=>
#P2' P1 y <= x <= P2' P1 y
Id = spdiagm(0 => ones(prod(comp_grid.n)) )
P =  Id[sortperm(vec(cameraman)),:]

# define constraint
constraint = Vector{SetIntersectionProjection.set_definitions}()

#histogram constraints
m_min     = P'*obs
m_max     = P'*obs
set_type  = "bounds"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP));

options.parallel       = false
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL);
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options);

@time (x4,log_PARSDMM) = PARSDMM(deepcopy(vec(cameraman)),AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

figure();
subplot(2,3,1);imshow(mandril);title("madril image")
subplot(2,3,2);imshow(cameraman);title("cameraman image")
subplot(2,3,3);imshow(reshape(x4,comp_grid.n));title("Cameraman image with histogram of mandril")
subplot(2,3,4);hist(vec(mandril));title("madril histogram")
subplot(2,3,5);hist(vec(cameraman));title("cameraman histogram")
subplot(2,3,6);hist(vec(x4));title("adjusted cameraman histogram")

## Constraint 5: Different constraints on different parameters ##
## as a toy example: different constraints on different channels of the RGB image

using Images
using TestImages
using SetIntersectionProjection
using PyPlot

#goal observe histogram of mandril and make the cameraman have the same histogram
mandril   = testimage("mandril")

#now set this up again, this time using the higher level tools in SetIntersectionProjection
comp_grid = compgrid((TF(1.0), TF(1.0)),(size(mandril,1), size(mandril,2)))

mandril   = channelview(mandril) #convert RGB to 3D array


# define constraints
constraint_channel_1 = Vector{SetIntersectionProjection.set_definitions}()
constraint_channel_2 = Vector{SetIntersectionProjection.set_definitions}()

#l2 constraint on gradient for channel 1
m_min     = 0.0
m_max     = 3.0
set_type  = "l2"
TD_OP     = "D2D"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint_channel_1, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP));

#tv constraint for channel 2
m_min     = 0.0
m_max     = 200.0
set_type  = "l1"
TD_OP     = "D2D"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint_channel_2, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP));

#no constraints on channel 3

#set up projectors
options.parallel       = false

(P_sub1,TD_OP1,set_Prop1) = setup_constraints(constraint_channel_1,comp_grid,options.FL);
(P_sub2,TD_OP2,set_Prop2) = setup_constraints(constraint_channel_2,comp_grid,options.FL);
(TD_OP1,AtA1,l1,y1)        = PARSDMM_precompute_distribute(TD_OP1,set_Prop1,comp_grid,options);
(TD_OP2,AtA2,l2,y2)        = PARSDMM_precompute_distribute(TD_OP2,set_Prop2,comp_grid,options);

#set up one function to project the separate RGB channels of the image onto different constraint sets
proj_channel_1 = x-> PARSDMM(x, AtA1, TD_OP1, set_Prop1, P_sub1, comp_grid, options)
proj_channel_2 = x-> PARSDMM(x, AtA2, TD_OP2, set_Prop2, P_sub2, comp_grid, options)

# Projection function
function prj(input)
    input = Float32.(input)
    output = deepcopy(input)
    (x1,dummy1,dummy2,dymmy3) = proj_channel_1(vec(input[1,:,:]))
    (x2,dummy1,dummy2,dymmy3) = proj_channel_2(vec(input[2,:,:]))
    
    output[1,:,:] .= reshape(x1,size(input)[2:3])
    output[2,:,:] .= reshape(x2,size(input)[2:3])
    return output
end

#project
mandril_projected = prj(mandril)


figure();
subplot(4,3,1);imshow(permutedims(mandril,(2,3,1)));title("madril image");colorbar()
subplot(4,3,4);imshow(mandril[1,:,:]);title("mandril ch1");colorbar()
subplot(4,3,5);imshow(mandril[2,:,:]);title("mandril ch2");colorbar()
subplot(4,3,6);imshow(mandril[3,:,:]);title("mandril ch3");colorbar()
subplot(4,3,7);imshow(permutedims(mandril_projected,(2,3,1)));title("projected madril image");colorbar()
subplot(4,3,10);imshow(mandril_projected[1,:,:]);title("projected mandril ch1");colorbar()
subplot(4,3,11);imshow(mandril_projected[2,:,:]);title("projected mandril ch2");colorbar()
subplot(4,3,12);imshow(mandril_projected[3,:,:]);title("projected mandril ch3");colorbar()