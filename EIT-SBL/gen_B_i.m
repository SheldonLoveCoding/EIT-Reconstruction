function B_i = gen_B_i(r_i, h)
    r = [];
    for i = 0:h-1
        r = [r, r_i^i];
    end
    B_i = toeplitz(r);
end