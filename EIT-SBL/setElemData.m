function elem_data = setElemData(sigma_values,selected_areas,fwd_model)
% % sigma_value: the value we want to set at FEM elements
% % selected_areas: the areas we want to set
% % fwd_model:the forward model
% % Author: Liu Xiaodong 2022/12/7
    if size(sigma_values,1) == size(selected_areas,1)
        elem_data = ones(size(fwd_model.elems,1),1);
        for i_area = 1:length(selected_areas)
            selected_area = selected_areas{i_area}; % (area_xmin,area_xmax; area_ymin,area_ymax)
            selected_elems = getElemIndex(selected_area,fwd_model);
            elem_data(selected_elems) = sigma_values(i_area);
        end
    else
        disp('The length of sigma_values is different from that of selected_areas.');
    end
end