
function [data_frame, data_track, frame_labels, track_labels, Tracks]= prep_tvseries_no_norm(path)


path2file= path;

load(path2file)

matFileName = whos('-regexp','features_');
temp_f=load(path2file,matFileName.name);
features_vgg=temp_f.(matFileName.name);
clear temp_f


track_labels={Tracks.labelid};
track_labels=([track_labels{:}])';
numFrames= {Tracks.numframes};
numFrames = ([numFrames{:}])';

%%%% Frame labels
frame_labels=[];
for i=1:length(numFrames)
    frame_labels=[frame_labels;repmat(track_labels(i),[numFrames(i),1])];
end

%%%%%%%%%%%%%%%%%
data_track=[];data_frame=[];


for i=1:size(features_vgg,3)
    frames=features_vgg(:,(1:numFrames(i)),i)';
    if size(frames,1)== 1
        data_track=[data_track;frames];
    else
        data_track=[data_track;mean(frames)];
    end
    data_frame=[data_frame;frames];
end



end
