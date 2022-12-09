function mu_x_list = gen_mu_x_list(mu_x, h, g)
    mu_x_list = {};
    for i_g = 1:g
        temp_index = [(i_g-1)*h + 1:i_g * h];
        mu_x_list{i_g} = mu_x(temp_index);
    end
end