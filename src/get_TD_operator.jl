export get_TD_operator

function get_TD_operator(comp_grid,TD_type,TF)
  # input: TD_type: string indicates which transform-domain operator to return
  #      : comp_grid: structure with computational grid information; comp_grid.n = [x z] number of gridpoints in each direction; comp_grid.d = [d1 d2] spacing between gridpoints in each direction.
  # output : TD_OP : transform domain operator as a sparse matrix or matrix-free object
  # Bas Peters



if TF==Float64
  TI=Int64
else
  TI=Int32
end

h1 = comp_grid.d[1];
h2 = comp_grid.d[2];
n1 = comp_grid.n[1];
n2 = comp_grid.n[2];

if length(comp_grid.n)==3 && comp_grid.n[3]>1 #use 3d version
  h3 = comp_grid.d[3]
  n3 = comp_grid.n[3]
  (D3D, D3x, D3y, D3z) = get_discrete_Grad(n1,n2,n3,h1,h2,h3);
  if TD_type=="TV" || TD_type=="D3D"# anisotropic TV operator
      TD_OP = D3D
      AtA_diag = false ; dense = false ; TD_n = (n1-1+n1+n1 , n2-1+n2+n2 , n3-1+n3+n3) ; banded=true
  elseif TD_type=="D_z" # vertical derivative operator
      TD_OP=D3z
      AtA_diag = false ; dense = false ; TD_n = (n1 , n2 , n3-1) ; banded=true
  elseif TD_type=="D_x" # lateral derivative operator
      TD_OP=D3x
      AtA_diag = false ; dense = false ; TD_n = (n1-1 , n2 , n3) ; banded=true
  elseif TD_type=="D_y" # lateral derivative operator
      TD_OP=D3y
      AtA_diag = false ; dense = false ; TD_n = (n1 , n2-1 , n3) ; banded=true
  elseif TD_type == "identity"
      TD_OP = speye(n1*n2*n3)
      AtA_diag = true ; dense = false ; TD_n = (n1,n2,n3) ; banded=true
  elseif TD_type == "DFT"
      error("currently no 3D DFT implemented, needs to be done soon")
    elseif TD_type == "DCT"
      error("currently no 3D DCT implemented, needs to be done soon")
  elseif TD_type == "curvelet" || TD_type == "Curvelet"
      error("currently no 3D DFT implemented, needs to be done soon")
  else
      error("provided an unknown transform domain operator. check function get_TD_operator(comp_grid,TD_type) for options")
  end
else #use 2d version
  (D2D, D2x, D2z) = get_discrete_Grad(n1,n2,h1,h2);
  if TD_type=="TV" || TD_type=="D2D"# anisotropic TV operator
      TD_OP = D2D
      AtA_diag = false ; dense = false ; TD_n = ( (n1-1)+n1 , n2+(n2-1) ) ; banded=true
  elseif TD_type=="D_z" # vertical derivative operator
      TD_OP=D2z
      AtA_diag = false ; dense = false ; TD_n = (n1 , n2-1) ; banded=true
  elseif TD_type=="D_x" # lateral derivative operator
      TD_OP=D2x
      AtA_diag = false ; dense = false ; TD_n = (n1-1 , n2) ; banded=true
  elseif TD_type == "identity"
    TD_OP = speye(n1*n2)
    AtA_diag = true ; dense = false ; TD_n = (n1,n2) ; banded=true
  elseif TD_type == "DCT"
    TD_OP=joDCT(convert(Int64,n1),convert(Int64,n2);planned=false,DDT=TF,RDT=TF)
    AtA_diag = true ; dense = true ; TD_n = (n1,n2) ; banded=false
  elseif TD_type == "DFT"
    TD_OP=joDFT(convert(Int64,n1),convert(Int64,n2);planned=false,DDT=TF,RDT=Complex{TF})
    AtA_diag = true ; dense = true ; TD_n = (n1,n2) ; banded=false
  elseif TD_type == "curvelet" || TD_type == "Curvelet"
    TD_OP = joCurvelet2D(convert(Int64,n1),convert(Int64,n2);DDT=TF,RDT=TF,real_crvlts=true,all_crvlts=true)
    AtA_diag = true ; dense = true ; TD_n = (0,0) ; banded=false
    else
      error("provided an unknown transform domain operator. check function get_TD_operator(comp_grid,TD_type) for options")
    end
end
if typeof(TD_OP)<:SparseMatrixCSC
  TD_OP=convert(SparseMatrixCSC{TF,TI},TD_OP)
end

return TD_OP, AtA_diag, dense, TD_n, banded
end
