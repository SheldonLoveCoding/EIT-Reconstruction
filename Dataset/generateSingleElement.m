function allPairElement = generareSingleElement(maxElementNum)
% % ��˳������PairElement��PairElement����������С��Ԫ
    allPairElement = zeros(maxElementNum, 2);
    for n = 1:maxElementNum
        temp = [n, n+1];
        allPairElement(n,:) = temp;
    end
end