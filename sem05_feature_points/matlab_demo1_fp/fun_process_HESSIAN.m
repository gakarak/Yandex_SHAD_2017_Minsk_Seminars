function [ ret ] = fun_process_HESSIAN( fimg, lstSigma)
%FUN_PROCESS_HESSIAN Summary of this function goes here
%   Detailed explanation goes here
    img=imread(fimg);
    if ~ismatrix(img)
        img=rgb2gray(img);
    end

    img=im2double(img);

    fltSize=128;
% %     lstSigma=4.0:1:22.0;
    lstFlt={};
    numSigma=numel(lstSigma);
    brdImg=zeros(size(img));
    brd2=floor(fltSize/2);
    brdImg(brd2:end-brd2, brd2:end-brd2)=1.0;
    imgFlt=zeros(size(img,1), size(img,2), numSigma);
    figure,
    for ii=1:numSigma
        tsigm=lstSigma(ii);
        tflt=fspecial('gaussian', [fltSize, fltSize], tsigm);
        imgf=(tsigm^2)*imfilter(img, tflt);
        [gx,gy]=gradient(imgf);
        [gxx,gxy]=gradient(gx);
        [gyx,gyy]=gradient(gy);
        %
        [fx,fy]=gradient(tflt);
        [fxx,fxy]=gradient(fx);
        [fyx,fyy]=gradient(fy);
        fzz=[[fxx,fxy];[fyx,fyy]];
        %
        subplot(2,3,1), imshow(gxx, []), title('Gxx');
        subplot(2,3,2), imshow(gxy, []), title('Gxy');
        subplot(2,3,3), imshow(gyx, []), title('Gyx');
        subplot(2,3,4), imshow(gyy, []), title('Gyy');
        subplot(2,3,5), imshow(fzz,[]),  title('Fxx, Fxy, Fyx, Fyy');
        detM=gxx.*gyy-gxy.*gyx;
        trM2=(gxx+gyy).^2;
        Rval=abs(detM-0.05*trM2).*brdImg;
        imgFlt(:,:,ii)=Rval;
        subplot(2,3,6), imshow(Rval, []), title(sprintf('sigma=%0.2f', tsigm));
        drawnow;
    end

    %%
    maxVal=max(imgFlt(:));
    for ii=1:numSigma
        timg=uint8(255*imgFlt(:,:,ii)/maxVal);
        imshow(timg), title(sprintf('sigma=%0.2f', lstSigma(ii)));
        drawnow;
        pause(0.1);
    end
    %%
    [SS,BB]=sort(imgFlt,3, 'descend');
    SS1=SS(:,:,1);
    BB1=lstSigma(BB(:,:,1));
    figure,
    subplot(2,2,1), imshow(BB1,[]); colorbar;
    subplot(2,2,2), imshow(SS1,[]); colorbar;
    thBB=max(SS1(:))*0.7;
    SSbin=SS1>thBB;
    BB2=BB1;
    BB2(SS1<thBB)=0;
    subplot(2,2,3), imshow(BB2,[]), colorbar;

    subplot(2,2,4), imshow(img,[]);
    % % rectangle('Position', [200,200,30,30]);
    % % rectangle('Position', [100,100,30,30]);
    blobs=bwconncomp(SSbin);
    for ii=1:blobs.NumObjects
        [tmax, tidx]=max(SS1(blobs.PixelIdxList{ii}));
        [rr,cc]=ind2sub(size(SS1), blobs.PixelIdxList{ii}(tidx));
        tsigm=BB1(rr,cc);
        disp([rr,cc,tsigm]);
        trad=tsigm*3;
        rectangle('Position', [cc-trad,rr-trad,2*trad,2*trad], 'Curvature', [1,1], 'EdgeColor', [1,0,0], 'LineWidth', 3);
        text(cc+trad, rr-trad, sprintf('s=%0.1f', tsigm), 'Color', 'red', 'FontSize', 14);
    end
    title(sprintf('size=%s',mat2str(size(img))));
    colorbar;
    ret=1;
end

