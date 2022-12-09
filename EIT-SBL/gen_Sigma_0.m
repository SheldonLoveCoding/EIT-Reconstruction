function Sigma_0 = gen_Sigma_0(B_list, gamma_i)
    if isempty(B_list)
        disp('B\_list is not exist! ');
    else
        g = size(B_list, 2);
        h = size(B_list{1}, 1);
        Sigma_0 = zeros(g*h, g*h);
        for i_g = 1:g
        	temp_index = [(i_g-1)*h + 1:i_g * h];
            Sigma_0(temp_index,temp_index) = gamma_i(i_g) * B_list{i_g};            
        end
    end
end