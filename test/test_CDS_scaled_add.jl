@testset "CDS_scaled_add_tests" begin

#test addition structured matrix
TI=Int64
TF=Float64;
n1=30;
n2=20;
N=n1*n2;
x=randn(TF,N);
y=zeros(TF,N);

#test structured matrix
#comp_grid = compgrid( (25, 25),( n1,n2 ) );
comp_grid = compgrid( (25, 25),( n1,n2 ) );
(TV_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",TF);
A=TV_OP'*TV_OP;
A=convert(SparseMatrixCSC{TF,TI},A);
(D_z, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",TF);
B=D_z'*D_z;
B=convert(SparseMatrixCSC{TF,TI},B);

C=A+B;

(R_A,offset_A)=mat2CDS(A)
(R_B,offset_B)=mat2CDS(B)
(R_C,offset_C)=mat2CDS(C)

rho=1.0;
CDS_scaled_add!(R_A,R_B,offset_A,offset_B,rho )

#test if CDS storage of the sum is the same as the sum of the CDS storage matrices
@test R_C==R_A
@test offset_C==offset_A

#test if it still works with a matrix-vector product
Cx_native = C*x;
fill!(y,0)
Cx_CDS = CDS_MVp(size(R_A,1),size(R_A,2),R_A,offset_A,x,y)
@test isapprox(Cx_native,Cx_CDS,rtol=10*eps())


#test random matrix
A=sprandn(100,100,0.01)
B=sprandn(100,100,0.01)
x=randn(100)
y=zeros(100)

A=A+B #make sure A has got all offsets of A and B

(R_A,offset_A)=mat2CDS(A)
(R_B,offset_B)=mat2CDS(B)

C=A.+B;
Cx_native = C*x;



rho=1.0;
CDS_scaled_add!(R_A,R_B,offset_A,offset_B,rho )

fill!(y,0)
Cx_CDS = CDS_MVp(size(R_A,1),size(R_A,2),R_A,offset_A,x,y)
@test isapprox(Cx_native,Cx_CDS,rtol=10*eps())

(R_C,offset_C)=mat2CDS(C)
@test R_C==R_A
@test offset_C==offset_A

end
