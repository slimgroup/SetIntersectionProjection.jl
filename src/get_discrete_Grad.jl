export get_discrete_Grad

function get_discrete_Grad{TF<:Real,TI<:Integer}(n1::TI,n2::TI,h1::TF,h2::TF)
# input: n1 : number of grid points in the lateral dimension (x)
#        n2 : number of grid points in the vertical (depth) dimension (z)
#        h1 : distance between grid points in the lateral dimension (x) (assumed constant)
#        h2 : distance between grid points in the vertical (depth) dimension (z) (assumed constant)
# output : TD_OP : transform domain operator as a sparse matrix or matrix-free object
# Bas Peters

    #define difference matrix D acting on vectorized model using Kronecker products
    Ix = speye(n1) #x
    Iz = speye(n2) #z
    Dx = spdiagm((ones(n1-1)*-1,ones(n1-1)*1),(0,1))./h1;
    Dz = spdiagm((ones(n2-1)*-1,ones(n2-1)*1),(0,1))./h2;
    #temp1 = spzeros(n1*(n2-1),n2*n1)
    #temp2 = spzeros(n2*(n1-1),n2*n1)
    D2z=kron(Dz,Ix)
    D2x=kron(Iz,Dx)
    D2D = vcat(D2z,D2x)
    #D = [Dx;Dz]
return D2D, D2x, D2z
end

function get_discrete_Grad{TF<:Real,TI<:Integer}(n1::TI,n2::TI,n3::TI,h1::TF,h2::TF,h3::TF)
# input: n1 : number of grid points in the lateral dimension (x)
#        n2 : number of grid points in the vertical (depth) dimension (z)
#        h1 : distance between grid points in the lateral dimension (x) (assumed constant)
#        h2 : distance between grid points in the vertical (depth) dimension (z) (assumed constant)
# output : TD_OP : transform domain operator as a sparse matrix or matrix-free object
# Bas Peters

    #define difference matrix D acting on vectorized model using Kronecker products
    Ix = speye(n1) #x
    Iy = speye(n2) #x
    Iz = speye(n3) #z
    Dx = spdiagm((ones(n1-1)*-1,ones(n1-1)*1),(0,1))./h1;
    Dy = spdiagm((ones(n2-1)*-1,ones(n2-1)*1),(0,1))./h2;
    Dz = spdiagm((ones(n3-1)*-1,ones(n3-1)*1),(0,1))./h3;
    #temp1 = spzeros(n1*(n2-1),n2*n1)
    #temp2 = spzeros(n2*(n1-1),n2*n1)
    D3z=kron(Dz,Iy,Ix)
    D3y=kron(Iz,Dy,Ix)
    D3x=kron(Iz,Iy,Dx)
    D3D = vcat(D3z,D3y,D3x)
    #D = [Dx;Dz]
return D3D, D3x, D3y, D3z
end
