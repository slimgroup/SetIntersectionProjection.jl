close all
clear

n1=1100;
n2=1100;
counter=1;

%https://map.openaerialmap.org/#/-73.212890625,-5.528510525692789,3/square/21001/59e62b7c3d6412ef722095bd?resolution=medium&_k=i7quni
%http://oin-hotosm.s3.amazonaws.com/571e5c0ccd0663bb003c31db/0/571e5dcb2b67227a79b4fb0d.tif
importfile_tif('S_A_1.tif')
temp=rgb2gray(S_A_1);
temp=imrotate(temp,-54);
temp=temp(2400:6000,3000:6700);
n_patches_1 = floor(size(temp,1)/n1);
n_patches_2 = floor(size(temp,2)/n2);
for i=1:n_patches_1
    for j=1:n_patches_2
        SA_patches(counter,:,:)=temp((i-1)*n1+1:i*n1,(j-1)*n2+1:j*n2);
        counter=counter+1;
    end
end

%https://map.openaerialmap.org/#/-73.212890625,-5.528510525692789,3/square/21001/59e62b7c3d6412ef722095c1?resolution=medium&_k=4c7vc1
%http://oin-hotosm.s3.amazonaws.com/571e5c0ccd0663bb003c31db/0/571e5dcb2b67227a79b4fb0e.tif
importfile_tif('S_A_2.tif')
temp=rgb2gray(S_A_2);
temp=imrotate(temp,-54);
temp=temp(2000:4900,4500:6800);
n_patches_1 = floor(size(temp,1)/n1);
n_patches_2 = floor(size(temp,2)/n2);
for i=1:n_patches_1
    for j=1:n_patches_2
        SA_patches(counter,:,:)=temp((i-1)*n1+1:i*n1,(j-1)*n2+1:j*n2);;
        counter=counter+1;
    end
end

%https://map.openaerialmap.org/#/-73.212890625,-5.528510525692789,3/square/21001/59e62b7c3d6412ef722095c1?resolution=medium&_k=4c7vc1
%http://oin-hotosm.s3.amazonaws.com/571e5c0ccd0663bb003c31db/0/571e5dcb2b67227a79b4fb0e.tif
importfile_tif('S_A_2.tif')
temp=rgb2gray(S_A_2);
temp=imrotate(temp,-54);
temp=temp(4509:6911,5676:7105);
n_patches_1 = floor(size(temp,1)/n1);
n_patches_2 = floor(size(temp,2)/n2);
for i=1:n_patches_1
    for j=1:n_patches_2
        SA_patches(counter,:,:)=temp((i-1)*n1+1:i*n1,(j-1)*n2+1:j*n2);
        counter=counter+1;
    end
end

%https://map.openaerialmap.org/#/-73.212890625,-5.528510525692789,3/square/21001/59e62b7c3d6412ef722095d7?resolution=medium&_k=rqbda8
%http://oin-hotosm.s3.amazonaws.com/571e5c0ccd0663bb003c31db/0/571e5dcb2b67227a79b4fb11.tif
importfile_tif('S_A_3.tif')
temp=rgb2gray(S_A_3);
temp=imrotate(temp,-54);
temp=temp(6000:7950,3000:6400);
n_patches_1 = floor(size(temp,1)/n1);
n_patches_2 = floor(size(temp,2)/n2);
for i=1:n_patches_1
    for j=1:n_patches_2
        SA_patches(counter,:,:)=temp((i-1)*n1+1:i*n1,(j-1)*n2+1:j*n2);
        counter=counter+1;
    end
end

%https://map.openaerialmap.org/#/-129.583740234375,-5.758105076529984,6/square/21001/59e62b7c3d6412ef722095e7?resolution=medium&_k=2x8f4o
%http://oin-hotosm.s3.amazonaws.com/571e5c0ccd0663bb003c31db/0/571e5dcb2b67227a79b4fb14.tif
importfile_tif('S_A_4.tif')
temp=rgb2gray(S_A_4);
temp=imrotate(temp,-54);
temp=temp(2000:3300,3000:6600);
n_patches_1 = floor(size(temp,1)/n1);
n_patches_2 = floor(size(temp,2)/n2);
for i=1:n_patches_1
    for j=1:n_patches_2
        SA_patches(counter,:,:)=temp((i-1)*n1+1:i*n1,(j-1)*n2+1:j*n2);
        counter=counter+1;
    end
end

%https://map.openaerialmap.org/#/-129.583740234375,-5.758105076529984,6/square/21001/59e62b7c3d6412ef722095ec?resolution=medium&_k=qkp4qa
%http://oin-hotosm.s3.amazonaws.com/571e5c0ccd0663bb003c31db/0/571e5dcb2b67227a79b4fb15.tif
importfile_tif('S_A_5.tif')
temp=rgb2gray(S_A_5);
temp=imrotate(temp,-54);
temp=temp(2000:4250,3000:6800);
n_patches_1 = floor(size(temp,1)/n1);
n_patches_2 = floor(size(temp,2)/n2);
for i=1:n_patches_1
    for j=1:n_patches_2
        SA_patches(counter,:,:)=temp((i-1)*n1+1:i*n1,(j-1)*n2+1:j*n2);
        counter=counter+1;
    end
end

%https://map.openaerialmap.org/#/-129.583740234375,-5.758105076529984,6/square/21001/59e62b7d3d6412ef72209640?resolution=medium&_k=4f0l7z
%http://oin-hotosm.s3.amazonaws.com/571e5c0ccd0663bb003c31db/0/571e5dcb2b67227a79b4fb23.tif
importfile_tif('S_A_6.tif')
temp=rgb2gray(S_A_6);
temp=imrotate(temp,-54);
temp=temp(2100:6500,3000:6750);
n_patches_1 = floor(size(temp,1)/n1);
n_patches_2 = floor(size(temp,2)/n2);
for i=1:n_patches_1
    for j=1:n_patches_2
        SA_patches(counter,:,:)=temp((i-1)*n1+1:i*n1,(j-1)*n2+1:j*n2);;
        counter=counter+1;
    end
end

SA_patches=SA_patches(randperm(39),:,:);
save('SA_patches','SA_patches')