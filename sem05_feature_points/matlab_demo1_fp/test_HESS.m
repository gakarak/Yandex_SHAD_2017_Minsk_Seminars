close all;
clear all;

fimg1='/home/ar/img/doge_test/doge3.jpg';
fimg2='/home/ar/img/doge_test/doge4.jpg';

lstSigma=4.0:1:22.0;
fun_process_HESSIAN(fimg1, lstSigma);
fun_process_HESSIAN(fimg2, lstSigma);

