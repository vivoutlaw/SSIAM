function [unknown_idx_] = remove_labels(labels,rm)

    unknown_idx_ = [];
    for i=1:size(rm,2)
        %indexes to remove unknown-class sample set
        unknown_idx2 = find(labels==rm(1,i))';
        unknown_idx_ = [unknown_idx_ unknown_idx2];
    end

    unknown_idx_ = unknown_idx_';
end