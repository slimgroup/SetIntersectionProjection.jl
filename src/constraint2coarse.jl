export constraint2coarse

function constraint2coarse(constraint,comp_grid,coarsening_factor)

  #constraint=deepcopy(constraint)

  #bound constraints: same as on fine grid

  #rank: same as on fine grid

  #cardinality in a transform-domain: same as on fine grid

  #define below what to do with norms (l1, l2 and nuclear)
  # will probably depend on constraints and interpolation type that is used
  # below we use simple heuristics, better methods may improve performance of the multilevel scheme


if length(comp_grid.n)==3 && comp_grid.n[3]>1 #use 3D

  # #for point-wise bound constraints in a transform-domain on a 3D grid if the transform-domain is a derivative operator:
  # for i=1:3
  #   if haskey(constraint,string("use_TD_bounds_",i)) && (constraint[string("use_TD_bounds_",i)]==true)
  #     if (constraint[string("TDB_operator_",i)] in ["D_x","D_y","D_z","DFT"])
  #       constraint[string("TD_LB_",i)]=constraint[string("TD_LB_",i)].*coarsening_factor
  #       constraint[string("TD_UB_",i)]=constraint[string("TD_UB_",i)].*coarsening_factor
  #     end
  #   end
  # end

  #for l1 norm in a transform-domain: on a 3D grid: ||.||_1(coarse)=||.||_1(fine)/(coarsening factor^3)
  for i=1:3
    if haskey(constraint,string("use_TD_l1_",i)) && constraint[string("use_TD_l1_",i)]==true
      constraint[string("TD_l1_sigma_",i)]=constraint[string("TD_l1_sigma_",i)]/(coarsening_factor^3)
    end
  end

  #for l2 norm in a transform-domain: on a 3D grid: ||.||_2(coarse)=||.||_2(fine)/sqrt(coarsening factor^3)
  for i=1:3
    if haskey(constraint,string("use_TD_l2_",i)) && constraint[string("use_TD_l2_",i)]==true
      constraint[string("TD_l2_sigma_",i)]=constraint[string("TD_l2_sigma_",i)]/(sqrt(coarsening_factor^3))
    end
  end

else #use 2D

  # #for point-wise bound constraints in a transform-domain on a 2D grid if the transform-domain is a derivative operator:
  # for i=1:3
  #   if haskey(constraint,string("use_TD_bounds_",i)) && (constraint[string("use_TD_bounds_",i)]==true)
  #     if (constraint[string("TDB_operator_",i)] in ["D_x","D_y","D_z","DFT"])
  #       constraint[string("TD_LB_",i)]=constraint[string("TD_LB_",i)].*coarsening_factor
  #       constraint[string("TD_UB_",i)]=constraint[string("TD_UB_",i)].*coarsening_factor
  #     end
  #   end
  # end

    #for l1 norm in a transform-domain: on a 2D grid: ||.||_1(coarse)=||.||_1(fine)/(coarsening factor^2)
    for i=1:3
      if haskey(constraint,string("use_TD_l1_",i)) && constraint[string("use_TD_l1_",i)]==true
        constraint[string("TD_l1_sigma_",i)]=constraint[string("TD_l1_sigma_",i)]/(coarsening_factor^2)
      end
    end

    #for l2 norm in a transform-domain: on a 2D grid: ||.||_2(coarse)=||.||_2(fine)/(coarsening factor)
    for i=1:3
      if haskey(constraint,string("use_TD_l2_",i)) && constraint[string("use_TD_l2_",i)]==true
        constraint[string("TD_l2_sigma_",i)]=constraint[string("TD_l2_sigma_",i)]/(coarsening_factor)
      end
    end

    #nuclear norm:
    if haskey(constraint,string("use_TD_nuclear_",i)) && constraint[string("use_TD_nuclear_",i)]== true
      constraint[string("TD_nuclear_norm_",i)]=constraint[string("TD_nuclear_norm_",i)]/2.7
    end

end #END if 2D or 3D

  #subspace constraint
  if haskey(constraint,"use_subspace") && constraint["use_subspace"]== true
    error("still need to define how to map subspace to a coarser grid in: function constraint2coarse")
    #maybe just use coarsening of matrix to start simple
  end



return constraint
end
