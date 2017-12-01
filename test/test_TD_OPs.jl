@testset "TD_OPs" begin
#test transform-domain operators (linear operators)

# 2D discrete gradient
  n1=9
  n2=6
  h1=0.99
  h2=1.123
  (D2D, D2x, D2z)=get_discrete_Grad(n1,n2,h1,h2)

  x=zeros(n1,n2) #test on a 'cross' image
  x[:,3]=1.0
  x[4,:]=1.0

  a1 = D2x*vec(x); a1=reshape(a1,n1-1,n2)
  a2 = D2z*vec(x); a2=reshape(a2,n1,n2-1)
  a3 = D2D*vec(x); a3a=a3[1:(n2-1)*n1]; a3b=a3[1+(n2-1)*n1:end];
  a3a = reshape(a3a,n1,n2-1)
  a3b = reshape(a3b,n1-1,n2)

  #this test depends on the values of h1 and h2, as well as the type of derivative
  @test a1==diff(x,1)./h1
  @test a2==diff(x,2)./h2

  #some more general tests:
  @test countnz(a1[:,3])==0
  for i in [1 2 4 5 6]
    @test a1[:,1]==a1[:,i]
  end

  #some more general tests:
  @test countnz(a2[4,:])==0
  for i in [1 2 3 5 6 7 8 9]
   @test a2[1,:]==a2[i,:]
  end

  @test a3a==a2
  @test a3b==a1

  # 3D discrete gradient
    n1=4
    n2=6
    n3=5
    h1=0.99
    h2=1.123
    h3=1.0
    (D3D, D3x, D3y, D3z)=get_discrete_Grad(n1,n2,n3,h1,h2,h3)

    x=zeros(n1,n2,n3) #test on a 'cross' image
    x[2,:,:]=1.0
    x[:,4,:]=1.0
    x[:,:,3]=1.0

    a1 = D3x*vec(x); a1=reshape(a1,n1-1,n2,n3)
    a2 = D3y*vec(x); a2=reshape(a2,n1,n2-1,n3)
    a3 = D3z*vec(x); a3=reshape(a3,n1,n2,n3-1)
    for i=1:n2
      @test a1[:,i,:]==diff(x[:,i,:],1)./h1
    end
    for i=1:n3
      @test a1[:,:,i]==diff(x[:,:,i],1)./h1
    end

    for i=1:n1
      @test a2[i,:,:]==diff(x[i,:,:],1)./h2
    end
    for i=1:n3
      @test a2[:,:,i]==diff(x[:,:,i],2)./h2
    end

    for i=1:n1
      @test a3[i,:,:]==diff(x[i,:,:],2)./h3
    end
    for i=1:n2
      @test a3[:,i,:]==diff(x[:,i,:],2)./h3
    end

end
