
function [ClustScores,bcube,nmi_score]=computescores(c,num_clust, labels)

if size(labels,1)==1
labels = labels';
end
if size(c,1)==1
c = c';
end


[wcp,predicted_label] =find_cluster_acc(c, num_clust, labels);
ClustScores =compute_clustScores( labels,predicted_label);
bcube=b3(labels,c) ; 
nmi_score = nmi(labels,c);
end