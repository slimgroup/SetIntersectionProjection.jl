export mat2CDS
function mat2CDS{TF<:Real,TI<:Integer}(A::SparseMatrixCSC{TF,TI})
"""
Convert sparse and square matrix A to compressed diagonal format (CDS)
Currently not multithreaded
"""


  # Find all nonzero diagonals
  (i,j,dummy) = findnz(A) #[i,j] = find(A);
  d           = sort(j-i)
  offset      = d[find(diff([-Inf;d]))] #d = d(find(diff([-inf; d])));
  offset      = convert(Array{TI,1},offset)

  # initialize
  (m,n) = size(A)
  p     = length(offset)
  R     = zeros(TF,m,p)

  #construct tall dense matrix with diagonals of A
  for i=1:length(offset)
    dA = diag(A,offset[i])
    if offset[i]>=0
      R[1:length(dA),i]=dA;
    elseif offset[i]<0
      R[end-length(dA)+1:end,i]=dA;
    end
  end

return R,offset
end
