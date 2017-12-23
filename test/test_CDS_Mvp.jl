@testset "CDS_MVp_mat2CDS_tests" begin
  TI=Int32
  TF=Float32;
  n1=30;
  n2=20;
  N=n1*n2;
  x=randn(TF,N);
  y=zeros(TF,N);

  #test structured matrix
  #comp_grid = compgrid( (25, 25),( n1,n2 ) );
  comp_grid = compgrid( (TF(25), TF(25)),( n1,n2 ) );
  (TV_OP, AtA_diag, dense, TD_n, banded)=get_TD_operator(comp_grid,"TV",TF);
  A=TV_OP'*TV_OP;
  A=convert(SparseMatrixCSC{TF,TI},A);

  Ax_native = A*x;

  (R,offset)=mat2CDS(A)
  fill!(y,0)
  Ax_CDS = CDS_MVp(size(R,1),size(R,2),R,offset,x,y)
  @test isapprox(Ax_native,Ax_CDS,rtol=10*eps())

  #test random matrix
  x=randn(1000);
  y=zeros(1000);
  A=sprandn(1000,1000,0.1)
  Ax_native = A*x;

  (R,offset)=mat2CDS(A)
  fill!(y,0)
  Ax_CDS = CDS_MVp(size(R,1),size(R,2),R,offset,x,y)
  @test isapprox(Ax_native,Ax_CDS,rtol=10*eps())

end
