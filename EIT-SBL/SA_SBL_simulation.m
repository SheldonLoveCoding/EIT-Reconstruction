clear
clc
close all;
% run 'D:\Project_CASIA\EIT\eidors-v3.9.1-ng\eidors\startup.m'

%% ��������������Ϣ��ǰ��ģ�� fwd_priori
% ����FEM
imdl_2d= mk_common_model('d2sc1',16);
if isfield(imdl_2d.fwd_model, 'coarse2fine')
    imdl_2d.fwd_model = rmfield(imdl_2d.fwd_model, 'coarse2fine');
end
% ���õ�������ģʽ
stim = mk_stim_patterns(16,1,'{ad}','{ad}',{},5);%���һ������ǿ��,��Ӱ�������ع�Ч��

% background
img_1 = mk_image(imdl_2d);
img_1.fwd_model.stimulation = stim;
img_1.fwd_solve.get_all_meas = 1;% �������FEM�õ��Ľ������
% figure(2);subplot(3,2,1);figure;show_fem(img_1);
baseline_0 = fwd_solve(img_1);

% with priori
img_priori = img_1;
% ������������ѡ��element������
% ����λ��Ҫ����img_1.fwd_model.nodes��ȷ��
selected_areas = {[-0.8,-0.5; -0.2,-0.1];...
                  [-0.2,0.5; -0.5,0.1];...
                  [-0.4,0.0; 0.4,0.6]};
sigma_values = [1.1;2.1;-1.1];
% selected_areas = {[-0.2,0.5; -0.5,0.1]};  .* 100
% sigma_values = [100];
elem_data = setElemData(sigma_values,selected_areas,img_priori.fwd_model);
img_priori.elem_data = elem_data;

img_priori.name = 'delta on one single element';
img_priori.fwd_model.stimulation = stim;
img_priori.fwd_solve.get_all_meas = 1;% �������FEM�õ��Ľ������
% figure(2);subplot(3,2,2);show_fem(img_priori);title('reference  priori \sigma');
% ������Ľ⣬data.meas�Ƿ���õ����ܱߵĲ����ĵ�ѹֵ
baseline_priori = fwd_solve(img_priori);
% volt = baseline_priori.meas;

inv2d= eidors_obj('inv_model', 'EIT inverse');
inv2d.reconst_type= 'difference'; %�����ֵ��ͼ
inv2d.jacobian_bkgnd.value = 1; %һ��0.7-1.2��Խ��߽�Խģ����ԽС�߽�Խ��ɢ����׼
inv2d.fwd_model = img_priori.fwd_model;
% Jacobian ����
img_bkgnd= calc_jacobian_bkgnd(inv2d);
Jaco = -calc_jacobian(img_bkgnd);
volt = Jaco * img_priori.elem_data;

%% SA-SBL
figure(100);
subplot(2,2,1);show_fem(img_priori);title('reference  priori \sigma');
SNR_list = [50,40,30];
for i_SNR = 1:3
    noise = genNoise(volt, SNR_list(i_SNR));
    volt_n = volt + noise;
    
    sigma = SA_SBL_EIT(Jaco, img_priori, volt_n, 50);

    img_SASBL.type = 'image';
    img_SASBL.fwd_model = img_priori.fwd_model;
    img_SASBL.name = 'solve by SA-SBL';
    img_SASBL.elem_data = sigma;
    subplot(2,2,i_SNR+1);show_fem(img_SASBL);title(['solve by SA-SBL, SNR=', num2str(SNR_list(i_SNR))]);
end
% show_slices(img_SASBL);