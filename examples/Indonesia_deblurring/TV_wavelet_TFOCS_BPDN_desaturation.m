clear
close all

%go to data directory
cd /data/slim/bpeters/SetIntersection_data_results/

load('m_evaluation.mat')
load('d_obs.mat')
addpath(genpath('TFOCS-master'))

for i=1:size(d_obs,1)
    %% Call the TFOCS solver
    m=vec(d_obs(i,:,:));
    n1 = size(d_obs,2);
    n2 = size(d_obs,3);
    maxI    = 255; % max pixel value
    mat = @(x) reshape(x,n1,n2);
    REWEIGHT    = false; % whether to do one iteration of reweighting or not

    mu              = 1e-5;
    er              = @(x) 1;%norm(x(:)-x_original(:))/norm(x_original(:));
    opts = [];
    opts.errFcn     = @(f,dual,primal) er(primal);
    opts.maxIts     = 300;
    opts.printEvery = 20;
    opts.tol        = 1e-5;
    opts.stopcrit   = 4;

    x0 = m;
    z0 = [];   % we don't have a good guess for the dual

    % build operators:
    %A is a mask for the non-saturated pixels, they are fixed +/- a small
    %accomodation for error
    ind_fix=find(m<125);
    ind_max=find(m==125);
    A=speye(n1*n2);
    A=A(ind_fix,:);
    W_tv        = linop_TV( [n1,n2] );
    normTV      = linop_TV( [n1,n2], [], 'norm' );

    b=A*m;
    EPS=norm(ones(length(b),1).*2,2);

    contOpts            = [];
    contOpts.maxIts     = 4;

    opts_copy = opts;
    opts_copy.normW2     = normTV^2;
    opts_copy.continuation  = true;
    l = m-2;
    u = m;
    u(ind_max)=255;
    opts_copy.box = [l,u];
    [x_tv,out_tv] = solver_sBPDN_W( A, W_tv, b, EPS, mu, x0(:), z0, opts_copy, contOpts);

    %% plot
    figure;imagesc(reshape(m,size(d_obs,2), size(d_obs,3)),[0 255]);colormap gray;colorbar; title('observed')
    figure;imagesc(reshape(x_tv,n1, n2),[0 255]);colormap gray;colorbar; title('reconstructed')
    %figure;imagesc(squeeze(m_evaluation(i,:,:)),[0 255]);colormap gray;colorbar; title('true')
    pause(1)
    x_TFOCS_tv_desaturation(i,:,:)=reshape(x_tv,n1,n2);
   
    SNR=@(in1,in2) 20*log10(norm(in1)/norm(in1-in2))
    SNR(vec(m_evaluation(i,:,:)),x_tv)
end

 save('x_TFOCS_tv_desaturation','x_TFOCS_tv_desaturation')

 %% Repeat but now with wavelet transform instead of TV
 
 
 for i=1:size(d_obs,1)
    %% Call the TFOCS solver
    m=vec(d_obs(i,:,:));
    n1 = size(d_obs,2);
    n2 = size(d_obs,3);
    maxI    = 255; % max pixel value
    mat = @(x) reshape(x,n1,n2);
    REWEIGHT    = false; % whether to do one iteration of reweighting or not

    mu              = 1e-5;
    er              = @(x) 1;%norm(x(:)-x_original(:))/norm(x_original(:));
    opts = [];
    opts.errFcn     = @(f,dual,primal) er(primal);
    opts.maxIts     = 300;
    opts.printEvery = 20;
    opts.tol        = 1e-5;
    opts.stopcrit   = 4;

    x0 = m;
    z0 = [];   % we don't have a good guess for the dual

    % build operators:
    %A is a mask for the non-saturated pixels, they are fixed +/- a small
    %accomodation for error
    ind_fix=find(m<125);
    ind_max=find(m==125);
    A=speye(n1*n2);
    A=A(ind_fix,:);
    W_wavelet   = linop_spot( opWavelet2(n1,n2,'Daubechies') );
    normWavelet      = linop_normest( W_wavelet );
    normW22     = normWavelet^2;

    b=A*m;
    EPS=norm(ones(length(b),1).*2,2);

    contOpts            = [];
    contOpts.maxIts     = 4;

    opts_copy = opts;
    opts_copy.normW2     = normW22;
    opts_copy.continuation  = true;
    l = m-2;
    u = m;
    u(ind_max)=255;
    opts_copy.box = [l,u];
    [x_wavelet,out_tv] = solver_sBPDN_W( A, W_wavelet, b, EPS, mu, x0(:), z0, opts_copy, contOpts);

    %% plot
    figure;imagesc(reshape(m,size(d_obs,2), size(d_obs,3)),[0 255]);colormap gray;colorbar; title('observed')
    figure;imagesc(reshape(x_wavelet,n1, n2),[0 255]);colormap gray;colorbar; title('reconstructed')
    %figure;imagesc(squeeze(m_evaluation(i,:,:)),[0 255]);colormap gray;colorbar; title('true')
    pause(1)
    x_TFOCS_Wavelet_desaturation(i,:,:)=reshape(x_wavelet,n1,n2);
   
    SNR=@(in1,in2) 20*log10(norm(in1)/norm(in1-in2))
    SNR(vec(m_evaluation(i,:,:)),x_wavelet)
end

 save('x_TFOCS_Wavelet_desaturation','x_TFOCS_Wavelet_desaturation')