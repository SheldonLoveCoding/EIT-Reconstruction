function [selected_elems] = getElemIndex(selected_areas,fwd_model)
% selected_areas:(area_xmin,area_xmax; area_ymin,area_ymax)
    nodes = fwd_model.nodes;
    elems = fwd_model.elems;
    node_index_x = (nodes(:,1) >=  selected_areas(1,1)) & (nodes(:,1) <=  selected_areas(1,2));
    node_index_y = (nodes(:,2) >=  selected_areas(2,1)) & (nodes(:,2) <=  selected_areas(2,2));
    node_index = find(node_index_x & node_index_y);
    elem_index = logical(zeros(size(elems,1),1));
    for n = 1:size(node_index,1)
        temp_index = (elems == node_index(n,1));
        temp_index = (temp_index(:,1) | temp_index(:,2) | temp_index(:,3));
        elem_index = elem_index | temp_index;
    end
    selected_elems = elem_index;
end