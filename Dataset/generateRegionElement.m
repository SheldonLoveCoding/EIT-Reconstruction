function [allRegionElement, maxRegionNum] = generateRegionElement(fn0_vector, regionNum, fwd)
% % fn0_vector ��֮ǰ�ֿ��nodes������������Чֵ
% % regionNum ��ȫ���������
% % allRegionElement �Ƿֿ���ÿһ��elements������
% % maxRegionNum ����Ч����ĸ���
    allRegionElement = {};
    maxRegionNum = 0;
    elem_node = fwd.elems;
    for n = 1:regionNum
        if ~(isempty(fn0_vector{n,1}))
            maxRegionNum = maxRegionNum + 1;
            node_index = fn0_vector{n,1};
            elem_index = logical(zeros(size(elem_node,1),1));
            for m = 1:size(node_index,1)
                temp_index = (elem_node == node_index(m,1));
                temp_index = (temp_index(:,1) | temp_index(:,2) | temp_index(:,3));
                elem_index = elem_index | temp_index;
            end
            allRegionElement{maxRegionNum, 1} = find(elem_index);
        end
    end
end