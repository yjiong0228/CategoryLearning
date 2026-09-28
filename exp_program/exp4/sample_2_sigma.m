function info_t = sample_2_sigma(feature_names, nTrial, prototype1, prototype2, sigma)
% sample probabilistic stimulus for Task2 (4D Gaussian prototype version)
%
% feature_names: ["green", "yellow", "pink", "blue"]
% nTrial: total number of stimuli
% prototype1, prototype2: 1x4 or 4x1 vectors, prototype positions in 4D space
% sigma: Gaussian width (controls generalization)
%
% Example:
%   prototype1 = [0.2, 0.8, 0.3, 0.6];
%   prototype2 = [0.7, 0.1, 0.9, 0.4];

    nFeature = length(feature_names);

    if nFeature ~= 4
        error('This version assumes exactly 4 features.');
    end

    % Ensure prototype vectors are row vectors
    prototype1 = prototype1(:)';
    prototype2 = prototype2(:)';

    if length(prototype1) ~= nFeature || length(prototype2) ~= nFeature
        error('prototype1 and prototype2 must each have length equal to number of features.');
    end

    if sigma <= 0
        error('sigma must be positive.');
    end

    info_t = table();

    % For each feature, divide [0,1] into nTrial intervals and sample one value from each interval
    for iF = 1:nFeature
        intervals = linspace(0, 1, nTrial + 1);
        feature_values = zeros(nTrial, 1);

        for i = 1:nTrial
            feature_values(i) = unifrnd(intervals(i), intervals(i+1));
        end

        % Randomize the order of values for this feature
        rand_idx = randperm(nTrial);
        feature_values = feature_values(rand_idx);

        info_t.(sprintf('feature%d_name', iF)) = repelem(feature_names(iF), nTrial)';
        info_t.(sprintf('feature%d', iF)) = feature_values;
    end

    % ===== Compute 4D Gaussian similarity =====
    % Extract the 4 feature columns into an nTrial x 4 numeric matrix
    X = zeros(nTrial, nFeature);
    for iF = 1:nFeature
        X(:, iF) = info_t.(sprintf('feature%d', iF));
    end

    % Squared Euclidean distance to each 4D prototype
    % Each row of X is one stimulus: [feature1, feature2, feature3, feature4]
    diff1 = X - prototype1;   % implicit expansion
    diff2 = X - prototype2;

    dist1_sq = sum(diff1.^2, 2);   % nTrial x 1
    dist2_sq = sum(diff2.^2, 2);   % nTrial x 1

    % Gaussian similarity
    sim1 = exp(-dist1_sq ./ (2 * sigma^2));
    sim2 = exp(-dist2_sq ./ (2 * sigma^2));

    % Normalize to probabilities
    prob1 = sim1 ./ (sim1 + sim2);
    prob2 = sim2 ./ (sim1 + sim2);

    % Save results before final row shuffle
    info_t.dist2_category1 = dist1_sq;
    info_t.dist2_category2 = dist2_sq;
    info_t.sim_category1   = sim1;
    info_t.sim_category2   = sim2;
    info_t.prob_category1  = prob1;
    info_t.prob_category2  = prob2;

    % Randomize overall order
    rand_idx = randperm(nTrial);
    info_t = info_t(rand_idx, :);

    % Add stimulus ID after shuffling
    nTotalTrial = height(info_t);
    info_t.stiID = (1:nTotalTrial)';

    % Reorder columns
    optional_order = {'stiID', ...
        'feature1_name','feature2_name','feature3_name','feature4_name', ...
        'feature1','feature2','feature3','feature4', ...
        'dist2_category1','dist2_category2', ...
        'sim_category1','sim_category2', ...
        'prob_category1','prob_category2'};

    existing_cols = intersect(optional_order, info_t.Properties.VariableNames, 'stable');
    info_t = info_t(:, existing_cols);

end