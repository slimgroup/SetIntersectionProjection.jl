export get_bound_constraints

"""
sets up bound constraints, based on global min and max values,
possibly specific bounds for the water layer in marine seismic imaging
Only returns a vector of the vectorized model size if necessary,
because projections onto bounds using a scalar is faster
"""
function get_bound_constraints(comp_grid,constraint)

if typeof(constraint["m_min"])<:Real && typeof(constraint["m_max"])<:Real && haskey(constraint,"water_depth")==false
  LB = constraint["m_min"]
  UB = constraint["m_max"]
else
  TF=typeof(constraint["m_min"][1])
  if length(comp_grid.n)==3 && comp_grid.n[3]>1 #use 3D version
    LB = ones(TF,comp_grid.n[1],comp_grid.n[2],comp_grid.n[3])*constraint["m_min"];
    UB = ones(TF,comp_grid.n[1],comp_grid.n[2],comp_grid.n[3])*constraint["m_max"];

    if haskey(constraint,"water_depth")==1
      water_bottom_index = convert(Int,floor(constraint["water_depth"]/comp_grid.d[3]));
      if water_bottom_index==0
        water_bottom_index=1
      end
      LB_water = ones(TF,comp_grid.n[1],comp_grid.n[2],comp_grid.n[3]) .* constraint["water_min"];
      UB_water = ones(TF,comp_grid.n[1],comp_grid.n[2],comp_grid.n[3]) .* constraint["water_max"];
      LB_water[:,:,water_bottom_index+1:end]=LB[:,:,water_bottom_index+1:end];
      UB_water[:,:,water_bottom_index+1:end]=UB[:,:,water_bottom_index+1:end];

      #now find tightest combination
      LB=max.(vec(LB),vec(LB_water))
      UB=min.(vec(UB),vec(UB_water))
    end #end setting water layer bounds 3D
  else # use 2D version
    LB = ones(TF,comp_grid.n[1],comp_grid.n[2])*constraint["m_min"]
    UB = ones(TF,comp_grid.n[1],comp_grid.n[2])*constraint["m_max"]

    if haskey(constraint,"water_depth")==true
      water_bottom_index = convert(Int,floor(constraint["water_depth"]/comp_grid.d[2]));
      if water_bottom_index==0
        water_bottom_index=1
      end
      LB_water = ones(TF,comp_grid.n[1],comp_grid.n[2]) .* constraint["water_min"];
      UB_water = ones(TF,comp_grid.n[1],comp_grid.n[2]) .* constraint["water_max"];
      LB_water[:,water_bottom_index+1:end]=LB[:,water_bottom_index+1:end];
      UB_water[:,water_bottom_index+1:end]=UB[:,water_bottom_index+1:end];

      #now find tightest combination
      LB=max.(vec(LB),vec(LB_water))
      UB=min.(vec(UB),vec(UB_water))
    end #end setting water layer bounds 2D
  end #end 2D part of the function
  LB = vec(LB);
  UB = vec(UB);
end

return LB,UB
end
