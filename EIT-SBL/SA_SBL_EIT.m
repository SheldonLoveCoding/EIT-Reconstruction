function sigma = SA_SBL_EIT(Jaco, img_priori, volt, gen_max)
    %%%%%%%%% INPUT %%%%%%%%%%%%%%
    beta = 0.05; h = 2;M = size(volt,1);
    N = size(img_priori.elem_data, 1);
    g = N-h+1;
    eplison_min = 5e-3; % 误差
%     gen_max = 10; % 最大迭代步数
    %%%%%%%%% INIT %%%%%%%%%%%%%%
    eplison = 1;gen = 0;
    mu_x = zeros(g*h, 1);
    Sigma_x = zeros(g*h, g*h);

    psi_list = {};
    phi_list = {};
    B_list = {};
    Psi = [];
    B = [];
    r_i = 0.9;

    for i_psi = 1:g
        psi_i = gen_psi_i(N, h, i_psi);
        B_i = gen_B_i(r_i, h);
        B_list{i_psi} = B_i;
        psi_list{i_psi} = psi_i;
        phi_list{i_psi} = Jaco * psi_i;
        Psi = [Psi, psi_i];
    end
    Phi = Jaco * Psi;

    gamma_i = ones(g,1);
    Sigma_0 = gen_Sigma_0(B_list, gamma_i);
    gamma_0 = 0.01 * sqrt(1/(N-1) * sum(abs(volt - mean(volt))));
    B_list_t = B_list;
    mu_x_old = mu_x;
    Titer_cost = [];
    while (eplison >= eplison_min) && (gen <= gen_max)
        tic
        Sigma_0 = gen_Sigma_0(B_list, gamma_i);
        Sigma_v = Phi * Sigma_0 * Phi' + gamma_0 * eye(size(Phi,1));
        Sigma_v_inv = inv(Sigma_v);
        % eq.15
        mu_x = Sigma_0 * Phi'* Sigma_v_inv * volt;
        % eq.16
        Sigma_x = Sigma_0 - Sigma_0 * Phi' * Sigma_v_inv * Phi * Sigma_0;
        % eq.22
        gamma_i_old = gamma_i;
        for i_g = 1:g
            if i_g == 1
                p = gamma_i_old(i_g) + beta * gamma_i_old(i_g + 1);
            elseif i_g == g
                p = gamma_i_old(i_g) + beta * gamma_i_old(i_g - 1);
            else
                p = gamma_i_old(i_g) + beta * gamma_i_old(i_g + 1) +...
                    beta * gamma_i_old(i_g - 1);
            end

            numerator = sqrt(B_list{i_g}) * phi_list{i_g}' * Sigma_v_inv * volt;
            numerator = norm(numerator);
            denumerator = (phi_list{i_g}' * Sigma_v_inv * phi_list{i_g} * B_list{i_g});
            denumerator = sqrt(trace(denumerator));
            gamma_i(i_g) = p * numerator / denumerator;            
        end
        % eq.23
        trace_sum = 0;
        Sigma_x_list = gen_Sigma_x_list(Sigma_x, h, g);
        for i_g = 1:g
            trace_sum = trace_sum + trace(Sigma_x_list{i_g} *...
                        phi_list{i_g}' * phi_list{i_g});
        end
        gamma_0 = (norm(volt - Phi * mu_x)^2 + trace_sum) / M;

        % eq.18
        mu_x_list = gen_mu_x_list(mu_x, h, g);
        B_list_new_t = {};
        for i_g = 1:g
            temp = Sigma_x_list{i_g} + mu_x_list{i_g} * mu_x_list{i_g}';
            B_list_new_t{i_g} = B_list_t{i_g} + temp / gamma_i(i_g); 
    %         B_list_new_t{i_g} = temp / gamma_i(i_g); 
        end
        B_list_t = B_list_new_t;
        % eq.21    
        for i_g = 1:g
            B_i_t = B_list_t{i_g};
            r_i_t = mean(diag(B_i_t, 1)) / mean(diag(B_i_t));
            r_i = sign(r_i_t) * min(abs(r_i_t), 0.99);
            B_i = gen_B_i(r_i, h);
            B_list{i_g} = B_i;
        end

        %loss
        eplison = norm(mu_x - mu_x_old) / norm(mu_x);
        mu_x_old = mu_x;
        disp(['eplison = ', num2str(eplison), ';  gen = ', num2str(gen)])
        gen = gen + 1;
        t = toc;
        Titer_cost = [Titer_cost; t];
    end

    sigma = Psi * mu_x;
    disp(['The time cost of each iteration is ', num2str(mean(Titer_cost)), 's'])
end