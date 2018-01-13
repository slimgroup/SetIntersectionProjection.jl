export setup_multi_level_PARSDMM

function setup_multi_level_PARSDMM{TF<:Real}(
                                   m        ::Vector{TF},
                                   n_levels ::Integer,
                                   coarsening_factor,
                                   comp_grid,
                                   constraint,
                                   options
                                   )

if TF==Float64
  TI = Int64
elseif TF==Float32
  TI = Int32
end
[@spawnat pid comp_grid for pid in workers()]

#detect 2D or 3D
if length(comp_grid.n)==3 && comp_grid.n[3]>1
  dim3 = true #indicate we are working in 3D
else
  dim3 = false
end

#allocate vectors of vectors and vectors of matrices to store required
#quantities for PARSDMM for each level
m_levels         = Vector{Vector{TF}}(n_levels)
if options.parallel==false
  TD_OP_levels   = Vector{Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joLinearFunction{TF,TF}}}}(n_levels)
else
  #TD_OP_levels   = Vector{DistributedArrays.DArray{SparseMatrixCSC{TF,TI},1,Array{SparseMatrixCSC{TF,TI},1}}}(n_levels)
  TD_OP_levels   = Vector{DistributedArrays.DArray{Union{JOLI.joLinearFunction{TF,TF}, SparseMatrixCSC{TF,TI}},1,Array{Union{JOLI.joLinearFunction{TF,TF}, SparseMatrixCSC{TF,TI}},1}}}(n_levels)

end
AtA_levels       = Vector{Vector{SparseMatrixCSC{TF,TI}}}(n_levels)
P_sub_levels     = Vector{Vector{Any}}(n_levels)
TD_Prop_levels   = Vector{Any}(n_levels)
comp_grid_levels = Vector{Any}(n_levels)

# set up constraints and matrices on coarse level
#1st level is the grid corresponding to the original model m (matrix or tensor)
(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,TF)
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

if typeof(AtA)==Vector{Array{TF,2}} #change AtA to CDS
  #AtA=convert(Vector{Array{TF,2}},AtA);
  AtA_levels       = Vector{Vector{Array{TF,2}}}(n_levels)
end

i=1
m_levels[i]         = m
TD_OP_levels[i]     = TD_OP
AtA_levels[i]       = AtA
P_sub_levels[i]     = P_sub
TD_Prop_levels[i]   = TD_Prop
comp_grid_levels[i] = comp_grid

for i=2:n_levels

  #new number of grid points on this level
  n = round.( comp_grid.n./coarsening_factor^(i-1) )

  #coarsen original model to new grids
  itp_m       = interpolate(reshape(m,comp_grid.n), BSpline(Constant()), OnGrid())
  if dim3
    m_level     = itp_m[linspace(1,comp_grid.n[1],n[1]), linspace(1,comp_grid.n[2],n[2]), linspace(1,comp_grid.n[3],n[3])]
  else
    m_level     = itp_m[linspace(1,comp_grid.n[1],n[1]), linspace(1,comp_grid.n[2],n[2])]
  end
  m_levels[i] = vec(m_level)

  #create computational grid information for this level
  comp_grid_levels[i]   = deepcopy(comp_grid)
  comp_grid_levels[i].n = size(m_level)
  comp_grid_levels[i].d = (comp_grid.n./n).*comp_grid.d

  # 'translate' constraints to coarser grids
  # the constraints are originally defined on the grid of the model on the finest grid.
  # We need to define the equivalent constraints on coarser grids
  constraint_level = constraint2coarse(constraint,comp_grid_levels[i],coarsening_factor)

  #set up constraints on new level
  (P_sub_l,TD_OP_l,TD_Prop_l) = setup_constraints(constraint_level,comp_grid_levels[i],TF)
  (TD_OP_l,AtA_l,dummy1,dummy2) = PARSDMM_precompute_distribute(m_levels[i],TD_OP_l,TD_Prop_l,comp_grid_levels[i],options)

  #save information
  TD_OP_levels[i]     = TD_OP_l
  AtA_levels[i]       = AtA_l
  P_sub_levels[i]     = P_sub_l
  TD_Prop_levels[i]   = TD_Prop_l

end #for loop

return m_levels, TD_OP_levels, AtA_levels, P_sub_levels, TD_Prop_levels, comp_grid_levels
end #end setup_multi_level_PARSDMM
