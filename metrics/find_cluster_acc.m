
function [wcp, predicted_label] =find_cluster_acc(idx,Num_clusters, identification_label)
% identification_label=zeros(size(truelabel));
% u=unique(truelabel);
% for k=1:length(u)
% in_=find(truelabel==u(k));
% identification_label(in_)=k;
% end



if min(idx)==0
    idx=idx+1;
end

acc= 0; wcp=0; predicted_label=zeros(length(idx),1);
for i=1:Num_clusters
    ind = find(idx==i);
    actual_cluster_i=identification_label(ind);
   % unique(actual_cluster_i);
    id = mode(actual_cluster_i);
    predicted_label(ind)= id;
    wcp=wcp+length(find(actual_cluster_i==id));
end

wcp=wcp/length(idx);
end