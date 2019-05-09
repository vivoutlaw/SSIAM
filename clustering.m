function [c] = clustering(features,Num_clusters,clustering_type)

if strcmp(clustering_type,'hac')
    
    % Maximum clusters is 6 now, since unknown class is discarded
    Z = linkage(features,'ward','euclidean');
    c = cluster(Z,'maxclust',Num_clusters);
    
elseif strcmp(clustering_type,'kmeans')
    
    % pool = parpool;                     % Invokes workers
    stream = RandStream('mlfg6331_64');
    options = statset('UseParallel',1,'UseSubstreams',1,...
        'Streams',stream);
    [c,C,sumd,D] = kmeans(features,Num_clusters,'Options',options,'MaxIter',10000,...
        'Display','final','Replicates',10);
    
    % delete(pool) 
end
end