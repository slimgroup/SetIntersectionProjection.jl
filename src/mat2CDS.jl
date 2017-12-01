export mat2CDS
function mat2CDS{TF<:Real,TI<:Integer}(A::SparseMatrixCSC{TF,TI})

  # Find all nonzero diagonals
  (i,j,dummy)=findnz(A) #[i,j] = find(A);
  d = sort(j-i);
  offset = d[find(diff([-Inf;d]))] #d = d(find(diff([-inf; d])));
  offset=convert(Array{TI,1},offset)

   (m,n) = size(A)
   p = length(offset)
   #R = zeros(TF,min(m,n),p);
   #if m>=n
     R=zeros(TF,m,p)
   #end
  # for k = 1:p
  #   if m >= n
  #     i = max(1,1+offset[k]):min(n,m+offset[k])
  #   else
  #     i = max(1,1-offset[k]):min(m,n-offset[k])
  #   end
  #   if ~isempty(i)
  #      R[i,k] = diag(A,d[k])
  #   end
  # end
  #
  # stencil_size=countnz(A[test_ind,:]);
  #
  # offset=Vector{TI}(stencil_size)
  # offset[1]=0
  # offset[2]=1
  # offset[3]=-1
  # offset[4]=n1
  # offset[5]=-n1
  # offset[6]=n1*n2
  # offset[7]=-n1*n2
  # offset=sort(offset)
  #
  #R=zeros(TF,N,stencil_size)
  for i=1:length(offset)
    dA=diag(A,offset[i])
    if offset[i]>=0
      R[1:length(dA),i]=dA;
    elseif offset[i]<0
      #prinln(size(R))
      #prinln(size(dA))
      R[end-length(dA)+1:end,i]=dA;
    # elseif offset[i]==0
    #   R[:,i]=dA;
    end
  end

return R,offset
end
