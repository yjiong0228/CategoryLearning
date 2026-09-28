

%% generate stimulus for Box learning task

nFeature = 5;
ruleID = 1;

subIDs = 500+(1:24);
for subID = subIDs
    rng(subID*2026);

    info = sample(nFeature, ruleID); % stimulus information
    writetable(info, ['Rule',num2str(ruleID),'_stimulus.csv']);

end

