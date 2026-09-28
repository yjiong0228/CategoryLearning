% %% generate mask
% 
% if ~exist('Masks', 'dir')
%     mkdir('Masks');
% end
% 
% nFeature = 4;
% N = 10;                 % 预生成数量
% maskSide = 800;         % 输出正方形边长（像素）——你可按需要调大
% tileSize = 50;          % 碎片块大小（像素）——越小越碎
% 
% img_feat_cell = cell(1, nFeature); 
% img_color = {'yellow','green','pink','blue'};
% for iFeature = 1:nFeature
%     img_path = ['feature-', img_color{iFeature}, '.png'];
%     [img_feat_rgb, ~, alpha] = imread(img_path, 'BackgroundColor', 'none');
%     img_feat_rgba = cat(3, img_feat_rgb, alpha);
%     img_feat_cell{iFeature} = img_feat_rgba;
% end
% 
% [img_center_rgb, ~, alpha] = imread('center.png', 'BackgroundColor', 'none');
% img_center = cat(3, img_center_rgb, alpha);
% 
% imgCell = {img_feat_cell{1}, img_feat_cell{2}, img_feat_cell{3}, img_feat_cell{4}, img_center};
% 
% 
% % ====== 生成 mask bank（存成 4D uint8: [H W 4]） ======
% masks = zeros(maskSide, maskSide, 4, 'uint8');
% bgRGBA = uint8([128 128 128 255]);
% 
% rng(202601);
% for i = 1:N
%     mask_image = make_scrambled_mask_from_images(imgCell, maskSide, tileSize, bgRGBA);
%     save([fullfile('Masks', num2str(i)), '.mat'], 'mask_image');
% end
% 


% %% generate structure settings for Task 2
% rng(202603);

% nFeature = 4;    % number of features in total
% nFtSelect = 3;   % number of features selected
% nSelect = nchoosek(nFeature, nFtSelect); % number of feature selection methods
% nSegmt = size(perms(1:nFtSelect),1);     % number of feature space segmentation methods

% nStructure = 24;

% structure1 = repelem(1:nSelect,nSegmt)';
% structure2 = repmat(1:nSegmt,1,nSelect)';

% structureID = (1:nStructure)';
% subID = [400 + (1:24)]'; 

% Task2_settings = table( ...
%     subID, ...
%     structureID, ...
%     structure1, ...
%     structure2, ...
%     'VariableNames', {'subID', 'structureID', 'structure1', 'structure2'} ...
% );

% Task2_settings.feature_names = arrayfun(@(s1, s2) covert(s1, s2), ...
%     Task2_settings.structure1, Task2_settings.structure2, 'UniformOutput', false);

% writetable(Task2_settings, 'Task2_settings.csv');


%% generate probabilistic stimulus for Task 3b learning
t2b1folder = fullfile('Task2', 'Stimulus', 'Block1');
if ~exist(t2b1folder, 'dir')
    mkdir(t2b1folder);
end

Task2_settings = readtable('Task2_settings.csv');

nTrial = 64;  
prototype1 = [0.25,0.5,0.5,0.5];
prototype2 = [0.75,0.5,0.5,0.5];
% sigma = 0.3;                   % smaller -> sharper posterior
beta = 10;

for idx = 1:length(Task2_settings.subID)
    
    subID = Task2_settings.subID(idx); 
    rng(subID*202603);

    feature_names = Task2_settings.feature_names(Task2_settings.subID == subID);
    feature_names = string(split(feature_names, ',')');
    
    info = sample_2_beta(feature_names, nTrial, prototype1, prototype2, beta);
    writetable(info, [fullfile(t2b1folder, num2str(subID)), '.csv']);

end


%%
function feature_names = covert(structure1, structure2)
    feature_names = strings(1, 4);

    % feature selection
    if structure1 == 1
        feature_names = ["green", "yellow", "pink", "blue"];
    elseif structure1 == 2
        feature_names = ["green", "yellow", "blue", "pink"];
    elseif structure1 == 3
        feature_names = ["green", "pink", "blue", "yellow"];
    elseif structure1 == 4
        feature_names = ["yellow", "pink", "blue", "green"];
    end

    % feature space segmentation
    if structure2 == 1
        feature_names = feature_names;
    elseif structure2 == 2
        feature_names = [feature_names(1), feature_names(3), feature_names(2), feature_names(4)];
    elseif structure2 == 3
        feature_names = [feature_names(2), feature_names(1), feature_names(3), feature_names(4)];
    elseif structure2 == 4
        feature_names = [feature_names(2), feature_names(3), feature_names(1), feature_names(4)];
    elseif structure2 == 5
        feature_names = [feature_names(3), feature_names(1), feature_names(2), feature_names(4)];
    elseif structure2 == 6
        feature_names = [feature_names(3), feature_names(2), feature_names(1), feature_names(4)];
    end

end
