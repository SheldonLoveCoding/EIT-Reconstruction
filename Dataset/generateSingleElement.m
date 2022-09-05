function allPairElement = generareSingleElement(maxElementNum)
% % 按顺序生成PairElement，PairElement用来代表最小单元
    allPairElement = zeros(maxElementNum, 2);
    for n = 1:maxElementNum
        temp = [n, n+1];
        allPairElement(n,:) = temp;
    end
end