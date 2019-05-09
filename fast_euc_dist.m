function d= fast_euc_dist(A)
%             orig_dist=pdist2(temp_gpu,temp_gpu,'euclidean'); % 'spearman'
s= sum(A.^2, 2);
d= real(sqrt( bsxfun(@plus, s, s') - 2*(A*A')));
end