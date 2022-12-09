clear
clc
close all;
% run 'D:\Project_CASIA\EIT\eidors-v3.9.1-ng\eidors\startup.m'

%% ��������������Ϣ��ǰ��ģ�� fwd_priori
% ����FEM
imdl_2d= mk_common_model('d2c',16);
if isfield(imdl_2d.fwd_model, 'coarse2fine')
    imdl_2d.fwd_model = rmfield(imdl_2d.fwd_model, 'coarse2fine');
end
% ���õ�������ģʽ
stim = mk_stim_patterns(16,1,'{ad}','{ad}',{},5);%���һ������ǿ��,��Ӱ�������ع�Ч��

% background
img_1 = mk_image(imdl_2d);
img_1.fwd_model.stimulation = stim;
img_1.fwd_solve.get_all_meas = 1;% �������FEM�õ��Ľ������
% figure(2);subplot(3,2,1);show_fem(img_1);
baseline_0 = fwd_solve(img_1);

img_calc_jaco.jacobian_bkgnd.value = 1; %һ��0.7-1.2��Խ��߽�Խģ����ԽС�߽�Խ��ɢ����׼
img_calc_jaco.fwd_model = img_1.fwd_model;
% Jacobian ����
img_bkgnd= calc_jacobian_bkgnd(img_calc_jaco);
Jaco = -calc_jacobian(img_bkgnd);
% volt = Jaco * img_priori.elem_data;

% SNR = 50
% noise = genNoise(volt, SNR);
% volt = volt + noise;
%% SA-SBL
path = 'D:\Project_CASIA\EIT-reconsturction\EIDORS_learning\';
txtnamelist = ['20221122-220336.txt'; '20221122-221224.txt';...
               '20220616-160827.txt'; '20220616-215726.txt';...
               '20221125-001837.txt'];
baseline_water = load([path, txtnamelist(1,:)]);
baseline_water = mean(baseline_water(1:100,:))';
data_water = load([path, txtnamelist(1,:)]);
data_water = data_water(:,1:208);
data_water = data_water .* mean(baseline_0.meas) ./ mean(baseline_water);
baseline_water = baseline_water .* mean(baseline_0.meas) ./ mean(baseline_water);

baseline_water = dataAdjust(baseline_water, baseline_0.meas);
%% ��diff����
frame_list = [53;511;1034];
for i_frame = 1:length(frame_list)
    volt_n = data_water(frame_list(i_frame),:)';
    volt_n = dataAdjust(volt_n, baseline_0.meas);
    
    sigma = SA_SBL_EIT(Jaco, img_1, volt_n, 1);
    img_SASBL.type = 'image';
    img_SASBL.fwd_model = img_1.fwd_model;
    img_SASBL.name = 'solve by SA-SBL';
    img_SASBL.elem_data = sigma;
    
    fig = figure(100);
    set(fig,'position',[100,375,1200,350]);
    subplot(1,2,1);
    cla;
    plot(baseline_0.meas);hold on;
    plot(volt_n);
    
    subplot(1,2,2);
    show_fem(img_SASBL);
    title(['SASBL-', num2str(frame_list(i_frame)), '   ',num2str(volt_n(1))])
    pause(1);
end
