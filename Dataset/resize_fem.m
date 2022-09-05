function [hotmap] = resize_fem(elem_value,node_map, fwd)
% % elem_value: 每一个element的值
% % node_map: 每一个node对应的索引
% % fwd 正向模型
% % hotmap 是resize后的图像

% % elem_map: 每一个块块对应的element的索引
    elem_map = node2elemmap(node_map, fwd);
    hotmap = zeros(size(elem_map));
    % 将每一个索引对应到降采样的块块里
    for n = 1:size(elem_map,1)
        for nn = 1:size(elem_map,2)
            if ~(isempty(elem_map{n,nn}))
                temp_index = elem_map{n,nn};
                hotmap(n,nn) = mean(elem_value(temp_index));
            end
        end
    end
    
    for i = 1:size(hotmap,1)
       for j = 1:size(hotmap,2)
           if(hotmap(i,j) == 0)
               hotmap(i,j) = NaN;
           end
       end
    end
    
end