function [ClustScores ]=compute_clustScores( true_label,predicted_label)

confusionMat=confusionmat(true_label,predicted_label);
ClustScores.wcp= trace(confusionMat)/sum(confusionMat(:));

recall =  diag(confusionMat)./sum(confusionMat,2);

precision = diag(confusionMat)./sum(confusionMat,1)';

f1Scores =  2*(precision.*recall)./(precision +recall);

precision(isnan(precision))=0; 
recall(isnan(recall))=0;
f1Scores(isnan(f1Scores))=0;


ClustScores.precision =mean(precision);
ClustScores.recall= mean(recall);
ClustScores.Fscore = mean(f1Scores);

end


