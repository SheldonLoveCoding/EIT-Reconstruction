clear
clc
close all;
% run 'D:\Project_CASIA\EIT\eidors-v3.9.1-ng\eidors\startup.m'

%% ��������������Ϣ��ǰ��ģ�� fwd_priori
% ����FEM
imdl_2d= mk_common_model('d2s',16);
if isfield(imdl_2d.fwd_model, 'coarse2fine')
    imdl_2d.fwd_model = rmfield(imdl_2d.fwd_model, 'coarse2fine');
end
% ���õ�������ģʽ
stim = mk_stim_patterns(16,1,'{ad}','{ad}',{},0.005);%���һ������ǿ��,��Ӱ�������ع�Ч��

% background
img_1 = mk_image(imdl_2d);
img_1.fwd_model.stimulation = stim;
img_1.fwd_solve.get_all_meas = 1;% �������FEM�õ��Ľ������
% figure(2);subplot(3,2,1);show_fem(img_1);
baseline_0 = fwd_solve(img_1);
baseline_0 = fwd_solve_1st_order(img_1);
% with priori
img_priori = img_1;
% ������������ѡ��element������
% ����λ��Ҫ����img_1.fwd_model.nodes��ȷ��
selected_area = [-0.8,-0.5; -0.2,-0.1]; % (area_xmin,area_xmax; area_ymin,area_ymax)
selected_elems = getElemIndex(selected_area,img_priori.fwd_model);
img_priori.elem_data(selected_elems) = 500;
selected_area = [-0.2,0.5; -0.5,0.1]; % (area_xmin,area_xmax; area_ymin,area_ymax)
selected_elems = getElemIndex(selected_area,img_priori.fwd_model);
img_priori.elem_data(selected_elems) = 500;
selected_area = [-0.4,0.0; 0.4,0.6]; % (area_xmin,area_xmax; area_ymin,area_ymax)
selected_elems = getElemIndex(selected_area,img_priori.fwd_model);
img_priori.elem_data(selected_elems) = 500;
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
%%%%%%%%% INPUT %%%%%%%%%%%%%%
beta = 0.05; h = 4;M = size(volt,1);
N = size(img_priori.elem_data, 1);
g = N-h+1;
eplison_min = 5e-3; % ���
gen_max = 50; % ����������
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
blkStartLoc = [];
for i_psi = 1:g
    psi_i = gen_psi_i(N, h, i_psi);
    B_i = gen_B_i(r_i, h);
    B_list{i_psi} = B_i;
    psi_list{i_psi} = psi_i;
    phi_list{i_psi} = Jaco * psi_i;
    Psi = [Psi, psi_i];
    blkStartLoc = [blkStartLoc; h*(i_psi-1)+1];
end
Phi = Jaco * Psi;
LearnLambda = 0;
Result = BSBL_EM(Phi, volt, blkStartLoc, LearnLambda, 'PRINT', 1, 'MAX_ITERS', 100);

% sigma = Psi * mu_x;
sigma = Psi * Result.x;
size(sigma);
figure;
img_SASBL.type = 'image';
img_SASBL.fwd_model = img_priori.fwd_model;
img_SASBL.name = 'solve by SA-SBL';
img_SASBL.elem_data = sigma;

subplot(1,2,1);show_slices(img_priori);title('reference  priori \sigma');
subplot(1,2,2);show_slices(img_SASBL);title('solve by SA-SBL');
