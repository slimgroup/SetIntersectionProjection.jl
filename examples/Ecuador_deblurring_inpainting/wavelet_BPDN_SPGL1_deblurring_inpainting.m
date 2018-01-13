clear
close all

%go to data directory
cd /data/slim/bpeters/SetIntersection_data_results/

load('m_evaluation.mat')
load('d_obs.mat')
load('FWD_OP.mat')

%add your SPGL1 folder to path
%addpath(genpath('/scratch/slim/bpeters/spgl1-master'))
%spgsetup %run spgl1 setup script

for i=1:size(d_obs,1)
    
    m=vec(d_obs(i,:,:));
    n1 = size(d_obs,2)+25;
    n2 = size(d_obs,3);
    
    options.iterations=300;
    options.optTol = 1e-5;
    % build operators:
    %TD_OP   = opWavelet2(n1,n2,'Haar');
    TD_OP   = opWavelet2(n1,n2,'Daubechies');
    %TD_OP = opCurvelet(n1,n2,max(1,ceil(log2(min(n1,n2)) - 3)),16,1);
    x0 = zeros(size(TD_OP,1),1);
    A           = opMatrix(FWD_OP)*TD_OP';
    
    b=m;
    sigma=norm(ones(size(A,1),1).*2,2);
    tau = 0;
    [x_W,r,g,info] = spgl1( A, b, tau, sigma, x0, options );
    
    x_W=TD_OP\x_W;
    x_W=reshape(x_W,n1, n2);
    %% plot
    figure;imagesc(reshape(m,size(d_obs,2), size(d_obs,3)),[0 255]);colormap gray;colorbar; title('observed')
    figure;imagesc(x_W,[0 255]);colormap gray;colorbar; title('reconstructed')
    %figure;imagesc(squeeze(m_evaluation(i,:,:)),[0 255]);colormap gray;colorbar; title('true')

    x_SPGL1_wavelet_save_SA(i,:,:)=x_W(50:end-50,50:end-50);
   
    SNR=@(in1,in2) 20*log10(norm(in1)/norm(in1-in2))
    SNR(vec(squeeze(m_evaluation(i,50:end-50,50:end-50))),vec(squeeze(x_SPGL1_wavelet_save_SA(i,:,:))))
end

 save('x_SPGL1_wavelet_save_SA','x_SPGL1_wavelet_save_SA')
