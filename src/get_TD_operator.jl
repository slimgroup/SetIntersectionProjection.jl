export get_TD_operator

"""
   input:
# Arguments
  - TD_type   : string indicates which transform-domain operator to return
  - comp_grid : structure with computational grid information; comp_grid.n = [n1 n2] number of gridpoints in each direction; comp_grid.d = [d1 d2] spacing between gridpoints in each direction.

   output
   - TD_OP   : transform domain operator as a sparse matrix or matrix-free object
"""
function get_TD_operator(comp_grid,TD_type,TF)

# if TF==Float64
#   TI=Int64
# else
#   TI=Int32
# end
TI=Int64

h1 = TF(comp_grid.d[1])
h2 = TF(comp_grid.d[2])
n1 = comp_grid.n[1]
n2 = comp_grid.n[2]

if length(comp_grid.n)==3 && comp_grid.n[3]>1 #use 3d version
  h3 = TF(comp_grid.d[3])
  n3 = comp_grid.n[3]
  #(D3D, D3x, D3y, D3z) = get_discrete_Grad(n1,n2,n3,h1,h2,h3);
  if TD_type=="TV" || TD_type=="D3D"# anisotropic TV operator
      TD_OP = get_discrete_Grad(n1,n2,n3,h1,h2,h3,TD_type)
      AtA_diag = false ; dense = false ; TD_n = (n1-1+n1+n1 , n2-1+n2+n2 , n3-1+n3+n3) ; banded=true
  elseif TD_type=="D_z" # vertical derivative operator
      TD_OP = get_discrete_Grad(n1,n2,n3,h1,h2,h3,TD_type)
      AtA_diag = false ; dense = false ; TD_n = (n1 , n2 , n3-1) ; banded=true
  elseif TD_type=="D_x" # lateral derivative operator
      TD_OP = get_discrete_Grad(n1,n2,n3,h1,h2,h3,TD_type)
      AtA_diag = false ; dense = false ; TD_n = (n1-1 , n2 , n3) ; banded=true
  elseif TD_type=="D_y" # lateral derivative operator
      TD_OP = get_discrete_Grad(n1,n2,n3,h1,h2,h3,TD_type)
      AtA_diag = false ; dense = false ; TD_n = (n1 , n2-1 , n3) ; banded=true
  elseif TD_type == "identity"
      TD_OP = SparseMatrixCSC{TF}(LinearAlgebra.I,n1*n2*n3,n1*n2*n3)
      AtA_diag = true ; dense = false ; TD_n = (n1,n2,n3) ; banded=true
  elseif TD_type == "DFT"
      TD_OP=joDFT(convert(Int64,n1),convert(Int64,n2),convert(Int64,n3);planned=false,DDT=TF,RDT=Complex{TF})
      AtA_diag = true ; dense = true ; TD_n = (n1,n2,n3) ; banded=false
    elseif TD_type == "DCT"
      TD_OP=joDCT(convert(Int64,n1),convert(Int64,n2),convert(Int64,n3);planned=false,DDT=TF,RDT=TF)
      AtA_diag = true ; dense = true ; TD_n = (n1,n2,n3) ; banded=false
  elseif TD_type == "curvelet"
      error("currently no 3D curvelet implemented, needs to be done soon")
    elseif TD_type == "wavelet"
      error("currently no 3D wavelet implemented, needs to be done soon")
  else
      error("provided an unknown transform domain operator. check function get_TD_operator(comp_grid,TD_type,TF) for options")
  end
else #use 2d version
  #(D2D, D2x, D2z) = get_discrete_Grad(n1,n2,h1,h2);
  if TD_type=="TV" || TD_type=="D2D"# anisotropic TV operator
      TD_OP = get_discrete_Grad(n1,n2,h1,h2,TD_type)
      AtA_diag = false ; dense = false ; TD_n = ( (n1-1)+n1 , n2+(n2-1) ) ; banded=true
  elseif TD_type=="D_z" # vertical derivative operator
      TD_OP = get_discrete_Grad(n1,n2,h1,h2,TD_type)
      AtA_diag = false ; dense = false ; TD_n = (n1 , n2-1) ; banded=true
  elseif TD_type=="D_x" # lateral derivative operator
      TD_OP = get_discrete_Grad(n1,n2,h1,h2,TD_type)
      AtA_diag = false ; dense = false ; TD_n = (n1-1 , n2) ; banded=true
  elseif TD_type=="D_xz" # lateral derivative operator composed with vertical derivative TD_OP=D_z*D_x
      D_x = get_discrete_Grad(n1,n2,h1,h2,"D_x")
      D_z = get_discrete_Grad(n1-1,n2,h1,h2,"D_z") #slightly different dimensions corresponding to the size of D_x*v (v=vector)
      TD_OP = D_z*D_x
      AtA_diag = false ; dense = false ; TD_n = (n1-1 , n2-1) ; banded=true
  elseif TD_type == "identity"
    TD_OP = SparseMatrixCSC{TF}(LinearAlgebra.I,n1*n2,n1*n2)
    AtA_diag = true ; dense = false ; TD_n = (n1,n2) ; banded=true
  elseif TD_type == "DCT"
    TD_OP=joDCT(convert(Int64,n1),convert(Int64,n2);planned=false,DDT=TF,RDT=TF)
    AtA_diag = true ; dense = true ; TD_n = (n1,n2) ; banded=false
  elseif TD_type == "DFT"
    TD_OP=joDFT(convert(Int64,n1),convert(Int64,n2);planned=false,DDT=TF,RDT=Complex{TF})
    AtA_diag = true ; dense = true ; TD_n = (n1,n2) ; banded=false
  elseif TD_type == "curvelet"
    TD_OP = joCurvelet2D(convert(Int64,n1),convert(Int64,n2);DDT=TF,RDT=TF,real_crvlts=true,all_crvlts=true)
    AtA_diag = true ; dense = true ; TD_n = (0,0) ; banded=false
  elseif TD_type == "wavelet"
     TD_OP = joDWT(n1,n2,wavelet(WT.db4);L=maxtransformlevels(min(n1,n2)),DDT=TF,RDT=TF)
     AtA_diag = true ; dense = true ; TD_n = (n1,n2) ; banded=false
  else
    error("provided an unknown transform domain operator. check function get_TD_operator(comp_grid,TD_type,TF) for options")
  end
end

return TD_OP, AtA_diag, dense, TD_n, banded
end
