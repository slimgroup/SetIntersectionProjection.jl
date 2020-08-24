@testset "Q_update_test" begin

#sparse matrix
  A=sprandn(100,100,0.1)
  B=Vector{SparseMatrixCSC{Float64,Int64}}(undef, 2)
  B[1]=sprandn(100,100,0.1)
  B[2]=sprandn(100,100,0.1)
  #make sure all nonzero diagonals in B are also in A, otherwise CDS doesnt work
  A=A+B[1]+B[2]
  A2=deepcopy(A)
  B2=deepcopy(B)
  A3=deepcopy(A)
  B3=deepcopy(B)
  maxit=100;
  p=2
  log_PARSDMM = log_type_PARSDMM(zeros(maxit,p-1),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),zeros(maxit),zeros(maxit),zeros(maxit,p),zeros(maxit,p),zeros(maxit),zeros(maxit),0.00,0.00,0.00,0.00,0.00,0.00,0.00);
  rho=Vector{Float64}(undef, 2)
  rho[1]=1.0
  rho[2]=1.0
  log_PARSDMM.rho[1,1]=2.0
  log_PARSDMM.rho[1,2]=3.0

  #add A = A+ 1.0*B[1] + 2.0*B[2]
  i=1
  ind_updated = findall(rho .!= log_PARSDMM.rho[i,:])  #::Vector{Integer}# locate changed rho index
  ind_updated = convert(Array{Int64,1},ind_updated)
  #use explicit solution:
    if isempty(ind_updated) == false
    for ii=1:length(ind_updated)
        A = A + ( B[ind_updated[ii]] )*( rho[ind_updated[ii]]-log_PARSDMM.rho[i,ind_updated[ii]] );
    end
    end
  #use Q_update:
    TI=Int64
    TF=Float64
    TD_Prop=set_properties(zeros(Bool, 99), zeros(Bool, 99), zeros(Bool, 99),
                           fill!(Vector{Tuple{TI,TI}}(undef, 99), (0, 0)),
                           fill!(Vector{Tuple{String,String, String, String}}(undef, 99), ("", "", "", "")),
                           zeros(Bool, 99),
                           Vector{Vector{TI}}(undef,99))
    A2=Q_update!(A2,B2,TD_Prop,rho,ind_updated,log_PARSDMM,i,[])

    x=randn(size(A,2))
    @test isapprox(A2*x,A*x,rtol=10*eps())

#sparse matrix in CDS format
  (R_A,offset_A)=mat2CDS(A3)
  (R_B1,offset_B1)=mat2CDS(B[1])
  (R_B2,offset_B2)=mat2CDS(B[2])
  R_B=Vector{Array{Float64,2}}(undef, 2)
  R_B[1]=R_B1
  R_B[2]=R_B2
  TD_Prop.AtA_offsets[1]=offset_B1
  TD_Prop.AtA_offsets[2]=offset_B2
  Q_update!(R_A,R_B,TD_Prop,rho,ind_updated,log_PARSDMM,i,offset_A)

  y=zeros(length(x))
  y=CDS_MVp(length(x),size(R_A,2),R_A,offset_A,x,y)
  @test isapprox(y,A*x,rtol=10*eps())



end
