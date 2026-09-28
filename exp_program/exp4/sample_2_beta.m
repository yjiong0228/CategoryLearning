function info_t = sample_2_beta(feature_names, nTrial, prototype1, prototype2, beta)
% sample probabilistic stimulus for Task2 (4D prototype + distance softmax)
%
% feature_names: ["green", "yellow", "pink", "blue"]
% nTrial: total number of stimuli
% prototype1, prototype2: 1x4 vectors, e.g. [0.2 0.4 0.6 0.8]
% beta: softmax parameter for posterior computation
%
% Output:
%   info_t contains:
%   - sampled 4D feature values
%   - distance to prototype1 / prototype2
%   - prob_category1 / prob_category2

    % -----------------------------
    % check inputs
    % -----------------------------
    if length(feature_names) ~= 4
        error('feature_names must contain exactly 4 feature names.');
    end

    if numel(prototype1) ~= 4 || numel(prototype2) ~= 4
        error('prototype1 and prototype2 must each be 1x4 vectors.');
    end

    if beta <= 0
        error('beta must be > 0.');
    end

    prototype1 = prototype1(:)';   % force row vector
    prototype2 = prototype2(:)';   % force row vector

    info_t = table();

    % -----------------------------
    % sample each feature from stratified intervals in [0,1]
    % -----------------------------
    for iF = 1:length(feature_names)
        intervals = linspace(0, 1, nTrial + 1);
        feature_values = zeros(nTrial, 1);

        for i = 1:nTrial
            feature_values(i) = unifrnd(intervals(i), intervals(i+1));
        end

        % randomize order within this feature
        rand_idx = randperm(nTrial);
        feature_values = feature_values(rand_idx);

        info_t.(sprintf('feature%d_name', iF)) = repelem(feature_names(iF), nTrial)';
        info_t.(sprintf('feature%d', iF)) = feature_values;
    end

    % -----------------------------
    % extract 4D stimulus matrix: nTrial x 4
    % -----------------------------
    X = [info_t.feature1, info_t.feature2, info_t.feature3, info_t.feature4];

    % -----------------------------
    % compute Euclidean distance to 4D prototypes
    % -----------------------------
    diff1 = X - prototype1;   % implicit expansion: nTrial x 4 - 1 x 4
    diff2 = X - prototype2;

    dist1 = sqrt(sum(diff1.^2, 2));
    dist2 = sqrt(sum(diff2.^2, 2));

    % -----------------------------
    % softmax over negative distances
    % -----------------------------
    logit1 = -beta * dist1;
    logit2 = -beta * dist2;

    maxlogit = max([logit1, logit2], [], 2);
    exp1 = exp(logit1 - maxlogit);
    exp2 = exp(logit2 - maxlogit);

    prob1 = exp1 ./ (exp1 + exp2);
    prob2 = exp2 ./ (exp1 + exp2);

    info_t.dist_category1 = dist1;
    info_t.dist_category2 = dist2;
    info_t.prob_category1 = prob1;
    info_t.prob_category2 = prob2;

    % -----------------------------
    % randomize overall trial order
    % -----------------------------
    rand_idx = randperm(nTrial);
    info_t = info_t(rand_idx, :);

    nTotalTrial = height(info_t);
    info_t.stiID = (1:nTotalTrial)';

    info_t.ambiguous = (info_t.prob_category1 >= 0.4 & info_t.prob_category1 <= 0.6) | (info_t.prob_category2 >= 0.4 & info_t.prob_category2 <= 0.6);

    % -----------------------------
    % reorder columns
    % -----------------------------
    optional_order = {'stiID', ...
        'feature1_name','feature2_name','feature3_name','feature4_name', ...
        'feature1','feature2','feature3','feature4', ...
        'dist_category1','dist_category2', ...
        'prob_category1','prob_category2', 'ambiguous'};

    existing_cols = intersect(optional_order, info_t.Properties.VariableNames, 'stable');
    info_t = info_t(:, existing_cols);

end