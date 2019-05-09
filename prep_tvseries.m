
function [data_frame, data_track, frame_labels, track_labels, Tracks]= prep_tvseries(path2file,path2InputFeats)

[~,NAME,~] = fileparts(path2file);
series = NAME(1:strfind(NAME,'_')-1);
season = str2num(NAME(strfind(NAME,'s')+1:strfind(NAME,'s')+2));
episode = str2num(NAME(strfind(NAME,'e')+1:strfind(NAME,'e')+2));
path2_pf8_tracks_linking = sprintf('%s/%s/%s_s%02de%02d.mat',...
                                path2InputFeats,series,series,season,episode);


load(path2file)
load(path2_pf8_tracks_linking)

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

%%%%%
% L2-Norm %%% both frame and track feats
ng=sqrt(sum(data_frame.^2,2));
data_frame=bsxfun(@rdivide,data_frame,ng);

nt=sqrt(sum(data_track.^2,2));
data_track=bsxfun(@rdivide,data_track,nt);
end