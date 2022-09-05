function [hotmap] = resize_fem(elem_value,node_map, fwd)
% % elem_value: ÿһ��element��ֵ
% % node_map: ÿһ��node��Ӧ������
% % fwd ����ģ��
% % hotmap ��resize���ͼ��

% % elem_map: ÿһ������Ӧ��element������
    elem_map = node2elemmap(node_map, fwd);
    hotmap = zeros(size(elem_map));
    % ��ÿһ��������Ӧ���������Ŀ����
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