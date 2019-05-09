function imdb = cnn_metriclearning_pdist2_setup_data(opts,imdb)

% series = 'bbt';
if strcmp(imdb.series,'bbt')
    labels_to_remove = 2;
end

load(sprintf('%s/%s/%s_s%02de%02d.mat',...
            opts.path2InputFeats,imdb.series,imdb.series,imdb.season,imdb.episode));

% Track ids, and track labels
track_labels={Tracks.labelid};
track_labels=([track_labels{:}])';

track_numframes={Tracks.numframes};
track_numframes= ([track_numframes{:}])';

track_trackerId={Tracks.trackerId};
track_trackerId= ([track_trackerId{:}])';

new_track_trackerId = [];
new_track_labels = [];
for i=1:size(track_numframes)
    temp_0 = repmat(track_trackerId(i),[1,track_numframes(i)]);
    new_track_trackerId = [new_track_trackerId, temp_0];
    
    temp_1 = repmat(track_labels(i),[1,track_numframes(i)]);
    new_track_labels = [new_track_labels, temp_1];
end
new_track_trackerId = new_track_trackerId';
new_track_labels = new_track_labels';

rm = [labels_to_remove, max(track_labels)];
remove_idx = [];
for i=1:size(rm,2)
    temp = find(new_track_labels==rm(i));
    remove_idx = [remove_idx, temp'];
end

all_indexes = 1:size(imdb.images.rawData.features_vgg,2);
idx_remaining = setdiff(all_indexes,remove_idx);
imdb.images.id = 1:size(idx_remaining,2);
imdb.images.labels = new_track_labels(idx_remaining)';
imdb.images.data = idx_remaining;
imdb.images.set = ones(1,size(imdb.images.id,2));
imdb.meta.sets = {'train', 'val', 'test'} ;

end