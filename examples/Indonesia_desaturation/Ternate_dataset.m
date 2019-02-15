clear
close all
temp=rgb2gray(a52434e40x2D95d20x2D4d540x2D81f10x2Db053ec739529);
 temp=temp(:,5000:end);

 low_res=temp(1:10:end,1:10:end);
 
  temp=temp(150*10:900*10,800*10:1300*10);
  n1=1500;
  n2=1250;
  
  counter=1;
  for i=1:5
      for j=1:4
          Ternate_patch(counter,:,:)=temp(n1*(i-1)+1:n1*i,n2*(j-1)+1:n2*j);
          counter=counter+1;
      end
  end
save('Ternate_patch','Ternate_patch')
  %Desa Sangaji Kota Ternate