function info_t = sample_2(feature_names, last, nTpCpB, ampercent)
% sample stimulus for Task2
% feature_names: with format like ["neck", "head", "leg", "tail"]
% nTpCpB: number of animals in each category in current block, with format of [a1,a2,b1,b2]


% features' range
feature_range = [0, 0.5, 1];

info_t = table();

% number of animals in each cube, cube_ID in each category: [13,57,26,48]
nperCube = [floor(nTpCpB(1)/2), floor(nTpCpB(3)/2), ceil(nTpCpB(1)/2), floor(nTpCpB(4)/2), ...
            floor(nTpCpB(2)/2), ceil(nTpCpB(3)/2), ceil(nTpCpB(2)/2), ceil(nTpCpB(4)/2)];  % in case nperCar is an odd number


% generate animal features, with format of (value of feature1, value of feature2, value of feature3)
% cube 1: (1, 1, 1); cube 2: (2, 1, 1); cube 3: (1, 2, 1); cube 4: (2, 2, 1);
% cube 5: (1, 1, 2); cube 6: (2, 1, 2); cube 7: (1, 2, 2); cube 8: (2, 2, 2)
nTrial = sum(nTpCpB);
info_t.stiID = last+(1:nTrial)'; % stiID
info_t.body_ori = 2*binornd(1,0.5,[nTrial,1])-1;  % body_ori

info_t.feature1_name = repelem(feature_names(1),nTrial)';
info_t.feature2_name = repelem(feature_names(2),nTrial)';
info_t.feature3_name = repelem(feature_names(3),nTrial)';
info_t.feature4_name = repelem(feature_names(4),nTrial)';


info_t.feature1 = [unifrnd(feature_range(1),feature_range(2),[nperCube(1),1]); 
                             unifrnd(feature_range(2),feature_range(3),[nperCube(2),1]);
                             unifrnd(feature_range(1),feature_range(2),[nperCube(3),1]); 
                             unifrnd(feature_range(2),feature_range(3),[nperCube(4),1]);
                             unifrnd(feature_range(1),feature_range(2),[nperCube(5),1]); 
                             unifrnd(feature_range(2),feature_range(3),[nperCube(6),1]);
                             unifrnd(feature_range(1),feature_range(2),[nperCube(7),1]); 
                             unifrnd(feature_range(2),feature_range(3),[nperCube(8),1])];

info_t.feature2 = [unifrnd(feature_range(1),feature_range(2),[nperCube(1),1]); 
                             unifrnd(feature_range(1),feature_range(2),[nperCube(2),1]);
                             unifrnd(feature_range(1),feature_range(2),[nperCube(3),1]); 
                             unifrnd(feature_range(1),feature_range(2),[nperCube(4),1]);
                             unifrnd(feature_range(2),feature_range(3),[nperCube(5),1]); 
                             unifrnd(feature_range(2),feature_range(3),[nperCube(6),1]);
                             unifrnd(feature_range(2),feature_range(3),[nperCube(7),1]); 
                             unifrnd(feature_range(2),feature_range(3),[nperCube(8),1])];

info_t.feature3 = [unifrnd(feature_range(1),feature_range(2),[nperCube(1),1]); 
                             unifrnd(feature_range(1),feature_range(2),[nperCube(2),1]);
                             unifrnd(feature_range(2),feature_range(3),[nperCube(3),1]); 
                             unifrnd(feature_range(2),feature_range(3),[nperCube(4),1]);
                             unifrnd(feature_range(1),feature_range(2),[nperCube(5),1]); 
                             unifrnd(feature_range(1),feature_range(2),[nperCube(6),1]);
                             unifrnd(feature_range(2),feature_range(3),[nperCube(7),1]); 
                             unifrnd(feature_range(2),feature_range(3),[nperCube(8),1])];

info_t.feature4 = unifrnd(feature_range(1),feature_range(3),[nTrial,1]);

info_t.category = repelem([1,3,1,4,2,3,2,4],nperCube)';  % category


% 计算模糊标签
ambiguous_fam = zeros(nTrial, 1); % 家族级模糊
ambiguous_spe = zeros(nTrial, 1); % 物种级模糊

% 计算模糊阈值（上下限）
threshold = [feature_range(2)-ampercent/2*(feature_range(3)-feature_range(1)), ...
    feature_range(2)+ampercent/2*(feature_range(3)-feature_range(1))];

for i = 1:nTrial
    % 家族级模糊：仅判断第一个特征
    if info_t.feature1(i) > threshold(1) && info_t.feature1(i) < threshold(2)
        ambiguous_fam(i) = 1;
    end
    
    % 物种级模糊：判断前三个特征任意一个
    if (info_t.feature1(i) > threshold(1) && info_t.feature1(i) < threshold(2)) || ...
       (info_t.feature2(i) > threshold(1) && info_t.feature2(i) < threshold(2)) || ...
       (info_t.feature3(i) > threshold(1) && info_t.feature3(i) < threshold(2))
        ambiguous_spe(i) = 1;
    end
end
info_t.ambiguous_fam = ambiguous_fam;
info_t.ambiguous_spe = ambiguous_spe;


optional_order = {'stiID','body_ori','feature1_name','feature2_name','feature3_name','feature4_name',...
    'feature1','feature2','feature3','feature4','category','ambiguous_fam','ambiguous_spe'};
info_t = info_t(:, optional_order);

end
