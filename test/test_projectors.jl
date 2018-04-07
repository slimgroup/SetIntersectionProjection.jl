@testset "projectors" begin

#test project_bounds! with scalar bounds
  x=randn(100)
  l=-0.11
  u=0.01;
  project_bounds!(x,l,u)
  @test maximum(x) <= u
  @test minimum(x) >= l

#test project_bounds! with vector bounds
  x=100.*randn(100)
  l=randn(100).-10
  u=randn(100).+10
  project_bounds!(x,l,u)
  for i=1:100
    @test x[i] <= u[i]
    @test x[i] >= l[i]
  end

#test project l1_Duchi!
  x=randn(100); tau=norm(x,1)*2; #see if projector returns untouched vector
  y=deepcopy(x)
  project_l1_Duchi!(x,tau)
  @test x == y

  x=randn(100); tau=norm(x,1)*0.234;
  project_l1_Duchi!(x,tau)
  @test isapprox(norm(x,1),tau,rtol=10*eps())

  #test project l1_Duchi! with a complex vector
  x=randn(100)+im*randn(100); tau=norm(x,1)*2; #see if projector returns untouched vector
  y=deepcopy(x)
  project_l1_Duchi!(x,tau)
  @test x == y

  x=randn(100)+im*randn(100); tau=norm(x,1)*0.234;
  project_l1_Duchi!(x,tau)
  @test isapprox(norm(x,1),tau,rtol=10*eps())

#test project_cardinality!

  #test vector mode
  x=randn(100)
  project_cardinality!(x,5)
  @test countnz(x)==5

  #test on closed form solution (vector)
  x=[0;0;1;2;3]
  project_cardinality!(x,2)
  @test x==[0;0;0;2;3]

  #test on closed form solution with negative numbers (vector)
  x=[0;0;-1;2;-3]
  project_cardinality!(x,2)
  @test x==[0;0;0;2;-3]

  #test mode that projects each row of a matrix separately
  X=randn(50,100)
  project_cardinality!(X,7,"x")
  for i=1:size(x,1)
    @test countnz(X[:,i])==7
  end
  X=randn(50,100)
  project_cardinality!(X,11,"z")
  for i=1:size(x,2)
    @test countnz(X[i,:])==11
  end

  #test mode that projects each fibre of a tensor separately (either in x,y or z direction)
  X=randn(50,60,30)
  project_cardinality!(X,7,"x_fiber")
  for i=1:size(x,2)
    for j=1:size(x,3)
      @test countnz(X[:,i,j])==7
    end
  end
  X=randn(50,60,30)
  project_cardinality!(X,6,"y_fiber")
  for i=1:size(x,1)
    for j=1:size(x,3)
      @test countnz(X[i,:,j])==6
    end
  end
  X=randn(50,60,30)
  project_cardinality!(X,4,"z_fiber")
  for i=1:size(x,1)
    for j=1:size(x,2)
      @test countnz(X[i,j,:])==4
    end
  end

# test project_l2!
  #test for sigma smaller than 2norm of vector
  x=randn(100)
  project_l2!(x,0.123)
  @test isapprox(norm(x,2),0.123,rtol=eps()*10)
  #test for sigma larger than 2norm of vector
  x=randn(100)
  y=deepcopy(x)
  project_l2!(x,1.234*norm(x,2))
  @test x==y


#test project_rank! (codes assumes input and output are vectors)
    X=randn(100,20)*randn(20,40) #randn(100,40) #test tall matrix
    grid_X=compgrid((1, 1),(100,40))
    Y=randn(40,50)*randn(50,100)#randn(40,100) #test flat matrix
    grid_Y=compgrid((1, 1),(40,100))
    Z=randn(100,30)*randn(30,100) #randn(100,100) #test square matrix
    grid_Z=compgrid((1, 1),(100,100))

    # test if input is returned untouched if rank is less than constraint
    XX=vec(deepcopy(X))
    YY=vec(deepcopy(Y))
    ZZ=vec(deepcopy(Z))
    X=vec(X)
    Y=vec(Y)
    Z=vec(Z)
    project_rank!(reshape(X,grid_X.n),30)
    project_rank!(reshape(Y,grid_Y.n),40)
    project_rank!(reshape(Z,grid_Z.n),35)
    @test isapprox(XX,X,rtol=20*eps())
    @test isapprox(YY,Y,rtol=20*eps())
    @test isapprox(ZZ,Z,rtol=20*eps())

    # test if output matrix has correct rank when constraint is smaller than matrix rank
    X=randn(100,40) #test tall matrix
    Y=randn(40,100) #test flat matrix
    Z=randn(100,100) #test square matrix
    X=vec(X); Y=vec(Y); Z=vec(Z)
    project_rank!(reshape(X,grid_X.n),29)
    project_rank!(reshape(Y,grid_Y.n),3)
    project_rank!(reshape(Z,grid_Z.n),64)
    X=reshape(X,grid_X.n[1],grid_X.n[2])
    Y=reshape(Y,grid_Y.n[1],grid_Y.n[2])
    Z=reshape(Z,grid_Z.n[1],grid_Z.n[2])
    r_X=rank(X); r_Y=rank(Y); r_Z=rank(Z);
    @test r_X==29
    @test r_Y==3
    @test r_Z==64

#test project nuclear! (codes assumes input and output are vectors)
  X=randn(100,40) #test tall matrix
  grid_X=compgrid((1, 1),(100,40))
  Y=randn(40,100) #test flat matrix
  grid_Y=compgrid((1, 1),(40,100))
  Z=randn(100,100) #test square matrix
  grid_Z=compgrid((1, 1),(100,100))

  nn_X = norm(svdvals(X),1)
  nn_Y = norm(svdvals(Y),1)
  nn_Z = norm(svdvals(Z),1)

  # test if input is returned untouched if nuclear norm is less than constraint
  X=vec(X)
  Y=vec(Y)
  Z=vec(Z)
  XX=deepcopy(X)
  YY=deepcopy(Y)
  ZZ=deepcopy(Z)
  project_nuclear!(reshape(X,grid_X.n),nn_X*1.1)
  project_nuclear!(reshape(Y,grid_Y.n),nn_Y*1.1)
  project_nuclear!(reshape(Z,grid_Z.n),nn_Z*1.1)
  @test isapprox(XX,X,rtol=20*eps())
  @test isapprox(YY,Y,rtol=20*eps())
  @test isapprox(ZZ,Z,rtol=20*eps())

  # test if output matrix has correct nuclear norm when constraint is smaller than matrix nuclear norm
  X=randn(100,40) #test tall matrix
  Y=randn(40,100) #test flat matrix
  Z=randn(100,100) #test square matrix
  nn_X = norm(svdvals(X),1)
  nn_Y = norm(svdvals(Y),1)
  nn_Z = norm(svdvals(Z),1)
  X=vec(X)
  Y=vec(Y)
  Z=vec(Z)
  project_nuclear!(reshape(X,grid_X.n),nn_X*0.5)
  project_nuclear!(reshape(Y,grid_Y.n),nn_Y*0.5)
  project_nuclear!(reshape(Z,grid_Z.n),nn_Z*0.5)
  X=reshape(X,grid_X.n[1],grid_X.n[2])
  Y=reshape(Y,grid_Y.n[1],grid_Y.n[2])
  Z=reshape(Z,grid_Z.n[1],grid_Z.n[2])
  nn_Xp = norm(svdvals(X),1)
  nn_Yp = norm(svdvals(Y),1)
  nn_Zp = norm(svdvals(Z),1)
  @test isapprox(nn_Xp,nn_X*0.5,rtol=20*eps())
  @test isapprox(nn_Yp,nn_Y*0.5,rtol=20*eps())
  @test isapprox(nn_Zp,nn_Z*0.5,rtol=20*eps())

#test project_subspace!
  #orthogonal subspace
  M=randn(100,50)
  F=svdfact(M)
  x=randn(100)
  y=deepcopy(x)
  project_subspace!(x,F.U,true)
  @test isapprox(x,F.U*(F.U'*x),rtol=eps()*10)

  #non-orthogonal subspace
  M=randn(100,50)
  x=randn(100)
  y=deepcopy(x)
  project_subspace!(x,M,true)
  @test isapprox(x,M*((M'*M)\(M'*x)),rtol=eps()*10)

end
