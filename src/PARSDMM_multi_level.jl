export PARSDMM_multi_level

function PARSDMM_multi_level{TF<:Real}(
                            m                 :: Vector{TF},#m_levels          ::Vector{Vector{TF}},
                            TD_OP_levels,
                            AtA_levels,        #::Union{Vector{Vector{SparseMatrixCSC{TF,TI}}},Vector{Vector{Array{TF,2}}}},
                            P_sub_levels      ::Vector{Vector{Any}},
                            TD_Prop_levels    ::Vector{Any},
                            comp_grid_levels  ::Vector{Any},
                            options,
                            x_ini=zeros(TF,prod(comp_grid_levels[end].n)) ::Vector{TF},
                            l_ini=[],
                            y_ini=[]
                            )

#::Union{Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joLinearFunction{TF,TF}}},DistributedArrays.DArray{Union{JOLI.joLinearFunction{TF,TF}, SparseMatrixCSC{TF,TI}},1,Array{Union{JOLI.joLinearFunction{TF,TF}, SparseMatrixCSC{TF,TI}},1}} },

n_levels = length(TD_OP_levels)

m_levels         = Vector{Vector{TF}}(n_levels) #allocate space for the model we will project at each level
m_levels[1]      = m

#interpolations/subsampling/coarsening is done with nearest neighbour
#from the "Interpolations" julia package. Change (BSpline(Constant())) for other
#types of interpolation
rho_orig = deepcopy(options.rho_ini)
#detect 2D or 3D
if length(comp_grid_levels[1].n)==3 && comp_grid_levels[1].n[3]>1
  dim3 = true #indicate we are working in 3D
else
  dim3 = false
end



#coarsen original model to new grids
for i = 2:n_levels
  itp_m       = interpolate(reshape(m,comp_grid_levels[1].n), BSpline(Constant()), OnGrid())
  if dim3
    m_level     = itp_m[linspace(1,comp_grid_levels[1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[1].n[2],comp_grid_levels[i].n[2]), linspace(1,comp_grid_levels[1].n[3],comp_grid_levels[i].n[3])]
  else
    m_level     = itp_m[linspace(1,comp_grid_levels[1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[1].n[2],comp_grid_levels[i].n[2])]
  end
  m_levels[i] = vec(m_level)
end

# for i=1:n_levels
#   println(minimum(m_levels[i]))
#   println(maximum(m_levels[i]))
#   println(sum(isnan.(m_levels[i])))
#   println(sum(isinf.(m_levels[i])))
# end

#start at coarsest grid (zero ini guess for x, y and l)
i=n_levels
#(x,log_PARSDMM,l,y) = compute_projection_intersection_PARSDMM(m_levels[i],AtA_levels[i],TD_OP_levels[i],TD_Prop_levels[i],P_sub_levels[i],comp_grid_levels[i],options,FL)
options.zero_ini_guess = true

#solve coarsest level more accurately
(x,log_PARSDMM,l,y) = PARSDMM(m_levels[i],AtA_levels[i],TD_OP_levels[i],TD_Prop_levels[i],P_sub_levels[i],comp_grid_levels[i],options,x_ini,l_ini,y_ini)
options.rho_ini = log_PARSDMM.rho[end,:]
for i=(n_levels-1):-1:1

  # resample x, l & y to a finer grid
  itp_x_level = interpolate(reshape(x,comp_grid_levels[i+1].n), BSpline(Constant()), OnGrid())
  if dim3
    x = itp_x_level[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2]), linspace(1,comp_grid_levels[i+1].n[3],comp_grid_levels[i].n[3])]
  else
    x = itp_x_level[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2])]
  end
  x = vec(x)

  #interpolate y and l
  if options.parallel==true #for now, gather -> interpolate -> distribute, this is inefficient and need to be updated
    l=convert(Vector{Vector{TF}},l)
    y=convert(Vector{Vector{TF}},y)
  end
  (l,y)=interpolate_y_l(l,y,TD_Prop_levels,comp_grid_levels,dim3,i)

  #account for distance squared term related to y and l: (always located at the end, and transform-domain operator = Identity)
  # itp_l   = interpolate(reshape(l[end],comp_grid_levels[i+1].n), BSpline(Constant()), OnGrid())
  # itp_y   = interpolate(reshape(y[end],comp_grid_levels[i+1].n), BSpline(Constant()), OnGrid())
  #
  # if dim3
  #   l_fine = itp_l[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2]), linspace(1,comp_grid_levels[i+1].n[3],comp_grid_levels[i].n[3])]
  #   y_fine = itp_y[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2]), linspace(1,comp_grid_levels[i+1].n[3],comp_grid_levels[i].n[3])]
  # else
  #   l_fine = itp_l[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2])]
  #   y_fine = itp_y[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2])]
  # end
  # l[end]=vec(l_fine)
  # y[end]=vec(y_fine)

  if options.parallel==true #for now, gather -> interpolate -> distribute, this is inefficient and need to be updated
    l=distribute(l)
    y=distribute(y)
  end

  # solve on a finer grid
  options.zero_ini_guess = false
  (x,log_PARSDMM,l,y) = PARSDMM(m_levels[i],AtA_levels[i],TD_OP_levels[i],TD_Prop_levels[i],P_sub_levels[i],comp_grid_levels[i],options,x,l,y)
  options.rho_ini = log_PARSDMM.rho[end,:]
  #(x,log_PARSDMM,l,y) = compute_projection_intersection_PARSDMM(m_levels[i],AtA_levels[i],TD_OP_levels[i],TD_Prop_levels[i],P_sub_levels[i],comp_grid_levels[i],options,FL,x,l,y)
end #end for loop over levels

options.rho_ini = rho_orig
return x, log_PARSDMM, l, y
end #end multi_level_PARSDMM
