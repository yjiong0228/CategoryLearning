%% generate structure for Task2

rng(202509);

nCategory = 4; % number of categories(species)
nFeature = 4; % number of features in total
nFtSelect = 3; % number of features selected
nSelect = nchoosek(nFeature, nFtSelect); % number of feature selection (3 out of 4) methods
nSegmt = size(perms(1:nFtSelect),1); % number of feature space segmentation methods

nSubperCon = 24;
nCondition = 3;
nSub = nSubperCon * nCondition;
condition = repelem([1,2,3],nSubperCon)'; % 1 = only learn family; 2 = only learn species; 3 = learn both

structure1 = repelem(1:nSelect,nSegmt)';
structure2 = repmat(1:nSegmt,1,nSelect)';

base = 1:nSubperCon;
subID = [base + 100, base + 200, base + 300]';

Task2_settings = table( ...
    subID, ...
    condition, ...
    repmat(structure1,nCondition,1), ...
    repmat(structure2,nCondition,1), ...
    'VariableNames', {'subID', 'condition', 'structure1', 'structure2'} ...
);

% Add the structure column using the covert function
Task2_settings.feature_names = arrayfun(@(s1, s2) covert(s1, s2), Task2_settings.structure1, Task2_settings.structure2, 'UniformOutput', false);

writetable(Task2_settings, 'Task2_settings.csv');



%% generate stimulus for Task2 block1

t2b1folder = fullfile('Task2', 'Stimulus_new', 'Block1');
if ~exist(t2b1folder, 'dir')
    mkdir(t2b1folder);
end

% task structure
Task2_settings = readtable('Task2_settings.csv'); % category structure, with format of [feature selection, feature space segmentation]

nCategory = 4;
nTpB = 64;
nTpCpB = [nTpB/nCategory, nTpB/nCategory, nTpB/nCategory, nTpB/nCategory]; % number of trials(stimulus) per category(species), with format of [a1,a2,b1,b2]

ampercent = 0.1; 

for idx = 1:length(Task2_settings.subID)
    
    subID = Task2_settings.subID(idx); 
    rng(subID*202510);

    feature_names = Task2_settings.feature_names(Task2_settings.subID == subID);
    feature_names = string(split(feature_names, ',')');

    info = sample_2(feature_names, 0, nTpCpB, ampercent); % stimulus information
    writetable(info, [fullfile(t2b1folder, num2str(subID)), '.csv']);

end


%%
function feature_names = covert(structure1, structure2)
    % 初始化 structure 为一个空值
    feature_names = strings(1, 4);

    % feature selection (根据 structure1 的值来选择特征顺序)
    if structure1 == 1
        feature_names = ["neck", "head", "leg", "tail"];
    elseif structure1 == 2
        feature_names = ["neck", "head", "tail", "leg"];
    elseif structure1 == 3
        feature_names = ["neck", "leg", "tail", "head"];
    elseif structure1 == 4
        feature_names = ["head", "leg", "tail", "neck"];
    end

    % feature space segmentation (根据 structure2 来调整特征顺序)
    if structure2 == 1
        feature_names = feature_names; % 保持不变
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

    % Final rearrangement (最终排序)
    feature_names = [feature_names(1), feature_names(3), feature_names(2), feature_names(4)];

end