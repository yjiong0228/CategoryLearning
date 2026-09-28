function real_features_t = transform_to_real(normalized_features, feature_range, angle)
    % Convert normalized features to real features
    nrows = size(normalized_features,1);

    % Get base features
    real_features = [repelem(feature_range.body_length,nrows)', angle, ...
        feature_range.green_length(1) + normalized_features.green_length * (feature_range.green_length(2) - feature_range.green_length(1)), ...
        feature_range.yellow_length(1) + normalized_features.yellow_length * (feature_range.yellow_length(2) - feature_range.yellow_length(1)), ...
        feature_range.pink_length(1) + normalized_features.pink_length * (feature_range.pink_length(2) - feature_range.pink_length(1)), ...
        feature_range.blue_length(1) + normalized_features.blue_length * (feature_range.blue_length(2) - feature_range.blue_length(1))];

    % Create table with appropriate variable names
    var_names = {'body_length','angle','green_length','yellow_length','pink_length','blue_length'};

    real_features_t = array2table(real_features, 'VariableNames', var_names);
end