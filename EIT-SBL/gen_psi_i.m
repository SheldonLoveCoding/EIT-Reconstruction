function psi_i = gen_psi_i(N, h, i)
    psi_i = [zeros(i-1, h); eye(h,h); zeros(N-i-h+1, h)];
end