close all;
clear all;

fimg1='/home/ar/img/doge_test/doge3.jpg';
fimg2='/home/ar/img/doge_test/doge4.jpg';


lstSigma=6.0:1:32.0;
fun_process_LOG(fimg1, lstSigma);
fun_process_LOG(fimg2, lstSigma);