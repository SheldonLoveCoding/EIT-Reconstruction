function [] = show_downsampled_fem(hotmap)
% % hotmap ÊÇresizeºóµÄÍ¼Ïñ
    if size(hotmap, 1) == 1
        hotmap = reshape(hotmap, 64, 64);
    end
    h = imagesc(hotmap); 
    grid on;
    colorbar('FontName', 'Times');
    set(gcf,'position',[50,550,1200,400]);
    set(h,'alphadata',~isnan(hotmap));
    set(gca,'xticklabel',[],...
         'yticklabel',[]);
end