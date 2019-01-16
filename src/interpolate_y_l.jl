export interpolate_y_l

"""
Interpolate l (vectors of Lagrangian multipliers in ADMM-based algorithms)
and y (auxiliary variable vectors) to finer grids
"""

function interpolate_y_l(
                        l                ::Vector{Vector{TF}},
                        y                ::Vector{Vector{TF}},
                        set_Prop_levels  ::Vector{Any},
                        comp_grid_levels ::Vector{Any},
                        dim3             ::Bool,
                        i                ::Integer
                        ) where {TF<:Real}

  for j=1:length(l) #loop over the various constraint sets


    if set_Prop_levels[i].tag[j][2] == "TV" || set_Prop_levels[i].tag[j][2]=="D2D" || set_Prop_levels[i].tag[j][2]=="D3D"#this tag contains the transform-domain operator type as a string
      if dim3 #use 3D
        p1e = (comp_grid_levels[i+1].n[1]-1)*comp_grid_levels[i+1].n[2]*comp_grid_levels[i+1].n[3]
        p2e = (comp_grid_levels[i+1].n[1]-1)*comp_grid_levels[i+1].n[2]*comp_grid_levels[i+1].n[3] + comp_grid_levels[i+1].n[1]*(comp_grid_levels[i+1].n[2]-1)*comp_grid_levels[i+1].n[3]

        y_part_1=y[j][1:p1e]
        y_part_2=y[j][p1e+1:p2e]
        y_part_3=y[j][p2e+1:end]

        l_part_1=l[j][1:p1e]
        l_part_2=l[j][p1e+1:p2e]
        l_part_3=l[j][p2e+1:end]

        itp_l_p1   = interpolate(reshape(l_part_1,comp_grid_levels[i+1].n[1]-1,comp_grid_levels[i+1].n[2],comp_grid_levels[i+1].n[3]), BSpline(Constant()), OnGrid())
        itp_y_p1   = interpolate(reshape(y_part_1,comp_grid_levels[i+1].n[1]-1,comp_grid_levels[i+1].n[2],comp_grid_levels[i+1].n[3]), BSpline(Constant()), OnGrid())

        itp_l_p2   = interpolate(reshape(l_part_2,comp_grid_levels[i+1].n[1],comp_grid_levels[i+1].n[2]-1,comp_grid_levels[i+1].n[3]), BSpline(Constant()), OnGrid())
        itp_y_p2   = interpolate(reshape(y_part_2,comp_grid_levels[i+1].n[1],comp_grid_levels[i+1].n[2]-1,comp_grid_levels[i+1].n[3]), BSpline(Constant()), OnGrid())

        itp_l_p3   = interpolate(reshape(l_part_3,comp_grid_levels[i+1].n[1],comp_grid_levels[i+1].n[2],comp_grid_levels[i+1].n[3]-1), BSpline(Constant()), OnGrid())
        itp_y_p3   = interpolate(reshape(y_part_3,comp_grid_levels[i+1].n[1],comp_grid_levels[i+1].n[2],comp_grid_levels[i+1].n[3]-1), BSpline(Constant()), OnGrid())

        l_1_fine = itp_l_p1[linspace(1,comp_grid_levels[i+1].n[1]-1,comp_grid_levels[i].n[1]-1), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2]), linspace(1,comp_grid_levels[i+1].n[3],comp_grid_levels[i].n[3])]
        y_1_fine = itp_y_p1[linspace(1,comp_grid_levels[i+1].n[1]-1,comp_grid_levels[i].n[1]-1), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2]), linspace(1,comp_grid_levels[i+1].n[3],comp_grid_levels[i].n[3])]

        l_2_fine = itp_l_p2[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2]-1,comp_grid_levels[i].n[2]-1), linspace(1,comp_grid_levels[i+1].n[3],comp_grid_levels[i].n[3])]
        y_2_fine = itp_y_p2[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2]-1,comp_grid_levels[i].n[2]-1), linspace(1,comp_grid_levels[i+1].n[3],comp_grid_levels[i].n[3])]

        l_3_fine = itp_l_p3[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2]), linspace(1,comp_grid_levels[i+1].n[3]-1,comp_grid_levels[i].n[3]-1)]
        y_3_fine = itp_y_p3[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2]), linspace(1,comp_grid_levels[i+1].n[3]-1,comp_grid_levels[i].n[3]-1)]

        l[j]=[vec(l_1_fine);vec(l_2_fine);vec(l_3_fine)]
        y[j]=[vec(y_1_fine);vec(y_2_fine);vec(y_3_fine)]
      else #use 2D version
        y_part_1=y[j][1:(comp_grid_levels[i+1].n[1]-1)*comp_grid_levels[i+1].n[2]]
        y_part_2=y[j][1+(comp_grid_levels[i+1].n[1]-1)*comp_grid_levels[i+1].n[2]:end]

        l_part_1=l[j][1:(comp_grid_levels[i+1].n[1]-1)*comp_grid_levels[i+1].n[2]]
        l_part_2=l[j][1+(comp_grid_levels[i+1].n[1]-1)*comp_grid_levels[i+1].n[2]:end]

        itp_l_p1   = interpolate(reshape(l_part_1,comp_grid_levels[i+1].n[1]-1,comp_grid_levels[i+1].n[2]), BSpline(Constant()), OnGrid())
        itp_y_p1   = interpolate(reshape(y_part_1,comp_grid_levels[i+1].n[1]-1,comp_grid_levels[i+1].n[2]), BSpline(Constant()), OnGrid())

        itp_l_p2   = interpolate(reshape(l_part_2,comp_grid_levels[i+1].n[1],comp_grid_levels[i+1].n[2]-1), BSpline(Constant()), OnGrid())
        itp_y_p2   = interpolate(reshape(y_part_2,comp_grid_levels[i+1].n[1],comp_grid_levels[i+1].n[2]-1), BSpline(Constant()), OnGrid())

        l_1_fine = itp_l_p1[linspace(1,comp_grid_levels[i+1].n[1]-1,comp_grid_levels[i].n[1]-1), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2])]
        y_1_fine = itp_y_p1[linspace(1,comp_grid_levels[i+1].n[1]-1,comp_grid_levels[i].n[1]-1), linspace(1,comp_grid_levels[i+1].n[2],comp_grid_levels[i].n[2])]

        l_2_fine = itp_l_p2[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2]-1,comp_grid_levels[i].n[2]-1)]
        y_2_fine = itp_y_p2[linspace(1,comp_grid_levels[i+1].n[1],comp_grid_levels[i].n[1]), linspace(1,comp_grid_levels[i+1].n[2]-1,comp_grid_levels[i].n[2]-1)]

        l[j]=[vec(l_1_fine);vec(l_2_fine)]
        y[j]=[vec(y_1_fine);vec(y_2_fine)]
      end
    else #for TD-operator is identity,D_z,D_x , etc

      #compute how many gridpoints more/less the transform-domain 'image' has
      #in each dimension compared to the model grid for x and m
      s = ( comp_grid_levels[i].n .- set_Prop_levels[i].TD_n[j] )
      #y and l have same dimensions
      itp_l   = interpolate(reshape(l[j],set_Prop_levels[i+1].TD_n[j]), BSpline(Constant()), OnGrid())
      itp_y   = interpolate(reshape(y[j],set_Prop_levels[i+1].TD_n[j]), BSpline(Constant()), OnGrid())

      if dim3
        l_fine = itp_l[linspace(1,set_Prop_levels[i+1].TD_n[j][1],comp_grid_levels[i].n[1]-s[1]), linspace(1,set_Prop_levels[i+1].TD_n[j][2],comp_grid_levels[i].n[2]-s[2]), linspace(1,set_Prop_levels[i+1].TD_n[j][3],comp_grid_levels[i].n[3]-s[3])]
        y_fine = itp_y[linspace(1,set_Prop_levels[i+1].TD_n[j][1],comp_grid_levels[i].n[1]-s[1]), linspace(1,set_Prop_levels[i+1].TD_n[j][2],comp_grid_levels[i].n[2]-s[2]), linspace(1,set_Prop_levels[i+1].TD_n[j][3],comp_grid_levels[i].n[3]-s[3])]
      else
        l_fine = itp_l[linspace(1,set_Prop_levels[i+1].TD_n[j][1],comp_grid_levels[i].n[1]-s[1]), linspace(1,set_Prop_levels[i+1].TD_n[j][2],comp_grid_levels[i].n[2]-s[2])]
        y_fine = itp_y[linspace(1,set_Prop_levels[i+1].TD_n[j][1],comp_grid_levels[i].n[1]-s[1]), linspace(1,set_Prop_levels[i+1].TD_n[j][2],comp_grid_levels[i].n[2]-s[2])]
      end
      l[j]=vec(l_fine)
      y[j]=vec(y_fine)
    end #end if flow for TD-operator type

  end # end loop over constraint sets

return l, y
end
