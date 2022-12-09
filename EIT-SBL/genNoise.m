function noise = genNoise(volt, SNR)
% % generate noise on voltage based on different SNR
    N = size(volt,1);  % number of the noise
    B = 20000;         %bandwidth
    P = (norm(volt,2).^2)/(power(10, SNR/10)); % the power of noise
    noise = wgn(N, 1 ,P, 'linear');
%     hist(noise, 50);
end