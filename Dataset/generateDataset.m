% % 这个程序是用来生成数据集的
clear                                                                                                                                                                                                                                                                                                                                                                                        clear
clc
close all;
% run 'D:\01Project\NI_EIT_Fitts\EIT\eidors-v3.9.1-ng\eidors\startup.m'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DATASET_TYPE = 2; % 1--pair dataset 2--region dataset
SAVE_FLAG = 1; % 0--not save 1--save 
delta_sigma = +0.5; % v1--- 1+delta_sigma 比 1 小 v2--- 1+delta_sigma 比 1 大
FIG_FLAG = 0; 
region_r = 0.08; % region 的半径
RESIZE_FLAG = 1;
LEN = 64; % resize后的边长
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2D FEM Model（正问题）
% 建立真实轮廓 FEM
% imdl_2d= mk_common_model('d2C',16);
load D:\01Project\NI_EIT_Fitts\EIT\MatlabForDrawEdgePicture\Data_001_shuzhi_Right
scale_Out=0.005; ind_Out=20; %0.005;% 轮廓范围等比例缩小，某些数值合适，原因不明
Out = [Data.Outer_Ring{1}.x   Data.Outer_Ring{1}.y]*scale_Out;%原始图片外围大圆
Out=roundn(Out,-2);Out = Out(1:ind_Out:end,:);
imdl_2d = ng_mk_extruded_model({0,{Out},{[4]},0.07},[16,1],[0.04 [0 0.15, 0, 0]]); %生成网格 4313 elements

% 根据索引进行分块（fn0_vector中存放的是nodes的索引）
node=imdl_2d.nodes;
x = node(:,1);y = node(:,2);
xmin = min(x);xmax = max(x);
ymin = min(y);ymax = max(y);
xn = linspace(0.46, 3.22, LEN);% 0.46:0.16:3.22;
yn = linspace(5.72, 1.76, LEN);% 5.72:-0.16:1.76;
[xn yn] = meshgrid(xn,yn);
r=region_r;
[saa sbb]=size(xn);
fn=1;
regionNum = 0;
fn0_vector = {};
fn0_map = {}; % 存的是node的索引
for i=1:saa
    for j=1:sbb
        regionNum=regionNum+1;  
%         %圆形
%         dis=sqrt((x-xn(i,j)).^2+(y-yn(i,j)).^2);
%         fn0=find(dis<=r); 
        %矩形
        fn11=find(x >= (xn(i,j)-r)); fn12=find(x <= (xn(i,j)+r));
        fn21=find(y >= (yn(i,j)-r)); fn22=find(y <= (yn(i,j)+r));
        fn1=intersect(fn11,fn12);fn2=intersect(fn21,fn22);
        fn0=intersect(fn1,fn2);
        fn0_vector{regionNum,1} = fn0;
        fn0_map{i,j} = fn0;
        fn=cat(1,fn,fn0);
    end
end
node_map = fn0_map;
% 设置电流激励模式
[stim, meas_sel] = mk_stim_patterns(16,1,'{ad}','{ad}',{},10);%最后一个电流强度,会影响最终重构效果
imdl_2d.stimulation = stim;
% 设置真实接触阻抗
real_z_contact = [0.9;0.9;0.6;0.7;0.6;0.7;0.7;0.6;...
                  0.6;0.6;0.6;0.7;0.7;0.7;0.7;0.8];
for i = 1:16
    imdl_2d.electrode(i).z_contact = real_z_contact(i);
end

% baseline
img_1 = mk_image(imdl_2d, 1);
img_1.fwd_solve.get_all_meas = 1;% 设置求解FEM得到的结果含义
baseline = fwd_solve_1st_order(img_1);

% delta
img_1_delta = img_1;

%% 逆问题的初始化
inv2d= eidors_obj('inv_model', 'EIT inverse');
inv2d.reconst_type= 'difference'; %用相对值画图
inv2d.jacobian_bkgnd.value = 1; %一般0.7-1.2，越大边界越模糊，越小边界越分散，不准
inv2d.fwd_model = img_1_delta.fwd_model;
% 求解（L2）
inv2d_l2 = inv2d;
% Jacobian 计算
img_bkgnd= calc_jacobian_bkgnd(inv2d_l2);
J = calc_jacobian(img_bkgnd);
% laplace
inv2d_l2.R_prior = @prior_laplace;
R = calc_R_prior(inv2d_l2);
RtR = R' * R;RtR_full = full(RtR);
lambda = 1e-9;
S = inv(J'*J + lambda * RtR) * J'*J;

%% 选择区域
if DATASET_TYPE == 1
    % pair 数据
    maxElementNum = size(imdl_2d.elems, 1);
    allPairElement = generateSingleElement(maxElementNum);
    allPairDataset = zeros(maxElementNum, maxElementNum + 208);
    loop_num = maxElementNum-1;
elseif DATASET_TYPE == 2
    % region 数据
    [allRegionElements, maxRegionNum] = generateRegionElement(fn0_vector, regionNum, inv2d.fwd_model);
    if RESIZE_FLAG == 1
        allRegionDataset = zeros(maxRegionNum, 208 + LEN*LEN);
    else
        allRegionDataset = zeros(maxRegionNum, 208 + size(imdl_2d.elems, 1));
    end
    loop_num = maxRegionNum;
end
%% 循环生成数据
tic
for s = 1:loop_num % floor(loop_num/2):floor(loop_num/2) 1:loop_num 
per_sample = [];
img_1_delta.elem_data(:) = 1;
if DATASET_TYPE == 1
   img_1_delta.elem_data(allPairElement(s,:)) = 1 + delta_sigma;
elseif DATASET_TYPE == 2
    img_1_delta.elem_data(allRegionElements{s,:}) = 1 + delta_sigma;
end

img_1_delta.name = 'delta on one single element';
img_1_delta.fwd_model.stimulation = stim;
img_1_delta.fwd_solve.get_all_meas = 1;% 设置求解FEM得到的结果含义


% 正问题的解，data.meas是仿真得到的周边的测量的电压值
data = fwd_solve_1st_order(img_1_delta);
td_volt = (data.meas - baseline.meas)';

% 逆问题 
index = [1: size(img_1_delta.elem_data,1)];
if(~isempty(strfind(imdl_2d.name, 'd2s')))
    index = mod([1: size(img_1_delta.elem_data,1)],2) == 1;
end
% img_1_reconstructed_SSA = img_1_reconstructed_map;
img_1_reconstructed_SSA.node_data = S * img_1_delta.elem_data(index');
img_1_reconstructed_SSA.fwd_model= inv2d.fwd_model;
img_1_reconstructed_SSA.name = 'solve by SSA';
img_1_reconstructed_SSA.type = 'image';
% resize
if RESIZE_FLAG == 1
    image_resized = resize_fem(img_1_reconstructed_SSA.node_data, node_map, inv2d.fwd_model);
    image_data = image_resized(:);
else
    image_data = img_1_reconstructed_SSA.node_data;
end
% 可视化
if FIG_FLAG == 1
    if RESIZE_FLAG == 0
        figure(1);subplot(1,2,1);
        show_fem(img_1_delta);title('reference  \delta\sigma');
        figure(1);subplot(1,2,2);
        show_fem(img_1_reconstructed_SSA);title('SSA');    
        pause(0.5);
    else
        figure(1);subplot(1,3,1);
        show_fem(img_1_delta);title('reference  \delta\sigma');
        figure(1);subplot(1,3,2);
        show_fem(img_1_reconstructed_SSA);title('SSA');
        figure(1);subplot(1,3,3);
        show_downsampled_fem(image_resized);title('resized');
        pause(0.5);
    end
end
if DATASET_TYPE == 1
    % pair 数据
    per_sample = [td_volt, image_data'];
    allPairDataset(s,:) = per_sample;
elseif DATASET_TYPE == 2
    % region 数据
    per_sample = [td_volt, image_data'];
    allRegionDataset(s,:) = per_sample;
end

if mod(s,10) == 0
disp(['This is ', num2str(s),'-th sample! Total number is ',num2str(loop_num),'.  ' num2str(s/loop_num*100),'% is done! ']);
end
end

if SAVE_FLAG && (DATASET_TYPE == 1)
    if delta_sigma < 0
        save(['allPairDataset_v1_',num2str(RESIZE_FLAG),'.mat'],'allPairDataset');
        disp('saved!!!');% v1--- 1+delta_sigma 比 1 小 v2--- 1+delta_sigma 比 1 大
    else delta_sigma > 0
        save(['allPairDataset_v2_',num2str(RESIZE_FLAG),'.mat'],'allPairDataset');disp('saved!!!');
    end
elseif SAVE_FLAG && (DATASET_TYPE == 2)
    if delta_sigma < 0
        save(['allRegionDataset_v1_',num2str(RESIZE_FLAG),'.mat'],'allRegionDataset');disp('saved!!!');
    else delta_sigma > 0
        save(['allRegionDataset_v2_',num2str(RESIZE_FLAG),'.mat'],'allRegionDataset');disp('saved!!!');
    end
end
toc