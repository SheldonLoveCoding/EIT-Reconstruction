function elem_map = node2elemmap(node_map, fwd)
    elem_map = {};
    elem_node = fwd.elems;
    row = size(node_map,1);
    col = size(node_map,2);
    for n = 1:row
        for nn = 1:col
            if ~(isempty(node_map{n,nn}))
                node_index = node_map{n,nn};
                elem_index = logical(zeros(size(elem_node,1),1));
                for m = 1:size(node_index,1)
                    temp_index = (elem_node == node_index(m,1));
                    temp_index = (temp_index(:,1) | temp_index(:,2) | temp_index(:,3));
                    elem_index = elem_index | temp_index;
                end
                elem_map{n, nn} = find(elem_index);
            else
                elem_map{n, nn} = [];
            end
        end
    end
    
end