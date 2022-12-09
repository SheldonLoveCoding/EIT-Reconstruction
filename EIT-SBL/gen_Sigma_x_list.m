function Sigma_x_list = gen_Sigma_x_list(Sigma_x, h, g)
    Sigma_x_list = {};
    for i_g = 1:g
        temp_index = [(i_g-1)*h + 1:i_g * h];
        Sigma_x_list{i_g} = Sigma_x(temp_index,temp_index);
    end
end