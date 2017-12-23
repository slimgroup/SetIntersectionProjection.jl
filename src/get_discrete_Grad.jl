export get_discrete_Grad

function get_discrete_Grad{TF<:Real}(n1,n2,h1::TF,h2::TF,TD_type::String)
# input: n1 : number of grid points in the lateral dimension (x)
#        n2 : number of grid points in the vertical (depth) dimension (z)
#        h1 : distance between grid points in the lateral dimension (x) (assumed constant)
#        h2 : distance between grid points in the vertical (depth) dimension (z) (assumed constant)
# output : TD_OP : transform domain operator as a sparse matrix or matrix-free object
# Bas Peters

if TF==Float64
  TI=Int64
else
  TI=Int32
end

    #define difference matrix D acting on vectorized model using Kronecker products
    Ix = speye(TF,n1) #x
    Iz = speye(TF,n2) #z
    Dx = spdiagm((ones(TF,n1-1)*-1,ones(TF,n1-1)*1),(0,1))./h1;
    Dz = spdiagm((ones(TF,n2-1)*-1,ones(TF,n2-1)*1),(0,1))./h2;

    Ix = convert(SparseMatrixCSC{TF,TI},Ix)
    Iz = convert(SparseMatrixCSC{TF,TI},Iz)
    Dx = convert(SparseMatrixCSC{TF,TI},Dx)
    Dz = convert(SparseMatrixCSC{TF,TI},Dz)

    if TD_type=="D_z"
      D_OP = kron(Dz,Ix) #D2z
    elseif TD_type=="D_x"
      D_OP = kron(Iz,Dx) #D2x
    elseif  TD_type=="TV" || TD_type=="D2D"
      D2z  = kron(Dz,Ix)
      D2x  = kron(Iz,Dx)
      D_OP = vcat(D2z,D2x) #D2D
    end
    #D = [Dx;Dz]
return D_OP
end

function get_discrete_Grad{TF<:Real}(n1,n2,n3,h1::TF,h2::TF,h3::TF,TD_type::String)
# input: n1 : number of grid points in the lateral dimension (x)
#        n2 : number of grid points in the vertical (depth) dimension (z)
#        h1 : distance between grid points in the lateral dimension (x) (assumed constant)
#        h2 : distance between grid points in the vertical (depth) dimension (z) (assumed constant)
# output : TD_OP : transform domain operator as a sparse matrix or matrix-free object
# Bas Peters

if TF==Float64
  TI=Int64
else
  TI=Int32
end

    #define difference matrix D acting on vectorized model using Kronecker products
    Ix = speye(TF,n1) #x
    Iy = speye(TF,n2) #x
    Iz = speye(TF,n3) #z
    Dx = spdiagm((ones(TF,n1-1)*-1,ones(TF,n1-1)*1),(0,1))./h1;
    Dy = spdiagm((ones(TF,n2-1)*-1,ones(TF,n2-1)*1),(0,1))./h2;
    Dz = spdiagm((ones(TF,n3-1)*-1,ones(TF,n3-1)*1),(0,1))./h3;

    Ix = convert(SparseMatrixCSC{TF,TI},Ix)
    Iy = convert(SparseMatrixCSC{TF,TI},Iy)
    Iz = convert(SparseMatrixCSC{TF,TI},Iz)
    Dx = convert(SparseMatrixCSC{TF,TI},Dx)
    Dy = convert(SparseMatrixCSC{TF,TI},Dy)
    Dz = convert(SparseMatrixCSC{TF,TI},Dz)

    if TD_type=="D_z"
      D_OP = kron(Dz,Iy,Ix)
    elseif TD_type=="D_y"
      D_OP = kron(Iz,Dy,Ix)
    elseif TD_type=="D_x"
      D_OP = kron(Iz,Iy,Dx)
    elseif TD_type=="TV" || TD_type=="D3D"
      D3z = kron(Dz,Iy,Ix)
      D3y = kron(Iz,Dy,Ix)
      D3x = kron(Iz,Iy,Dx)
      D_OP = vcat(D3z,D3y,D3x)
    end

return D_OP#D3D, D3x, D3y, D3z
end
