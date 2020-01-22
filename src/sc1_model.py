from warnings import filterwarnings
filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from load_data import sc1_phenotype, sc1_outcome, sc1_rna_data


def sc1_model(rna_data):
    """We collected 20 gene features based on literature of related research,
    and divided into two groups by their biological function.
    
    ['PLAT', 'DUSP5', 'VHL', 'DUSP10', 'COL5A1', 'IL13RA2',
     'CCR4', 'XCL1', 'STAT4', 'MMP9', 'JUN', 'BRCA2', 'MET'] is adverse to patient's
    survival if gene has high expression 
    
    ['SHC2', 'CREBBP', 'FGF22', 'EP300', 'ZNF346', 'FOXO4', 
    'AMH', 'FGF20', 'PTEN', 'CDKN2B', 'CDKN2A', 'CCL19', 'PTCH1'] is adverse to patient's
    survival if gene has low expression 
    Parameters
    ----------
    rna_data:
        pandas DataFrame, Gene expression profiles.
    Returns
    -------
    score:
        pandas DataFrame, survival score.
    """
    pos_pair = ['PLAT', 'DUSP5', 'VHL', 'DUSP10', 'COL5A1', 
                'IL13RA2', 'CCR4', 'XCL1', 'STAT4', 'MMP9', 
                'JUN', 'BRCA2', 'MET']
    neg_pair =['SHC2', 'CREBBP', 'FGF22', 'EP300', 'ZNF346', 
                'FOXO4', 'AMH', 'FGF20', 'PTEN', 'CDKN2B', 
                'CDKN2A', 'CCL19', 'PTCH1']

    rna_data['score'] = rna_data[pos_pair].mean(axis=1)-rna_data[neg_pair].mean(axis=1)
    return rna_data[['score']]


def cutoff_youdens_j(data):
    """Calculate Youden's J statistic for selecting the optimum cut-off point of score in
    train dataset.
        J = sensitivity + specificity -1
    Parameters
    ----------
    data:
        pandas DataFrame, score and SURVIVAL_STATUS must in columns.
    Returns
    -------
    j_ordered:
        pandas DataFrame, Youden's J statistic.
    """
    
    assert 'score' in data.columns
    assert 'SURVIVAL_STATUS' in data.columns
    score = data[['SURVIVAL_STATUS','score']].copy()
    fpr, tpr, threshold = roc_curve(score['SURVIVAL_STATUS'].values.astype(int), score['score'].values)
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,threshold))
    j_ordered = pd.DataFrame(columns=['youden_j','threshold'],data=j_ordered)
    return j_ordered.sort_values('youden_j',ascending=False)

if __name__ == '__main__':

    survival_score = sc1_model(sc1_rna_data)
    result = pd.merge(sc1_outcome,survival_score,left_index=True,right_index=True)
    threshold = cutoff_youdens_j(result)['threshold'].values[0]
    result['predict_label'] = result['score'].apply(lambda x:1 if x>threshold else 0)
    
    # Model Performance
    [[tn,fp],[fn,tp]] = confusion_matrix(result['SURVIVAL_STATUS'].values, 
                                         result['predict_label'].values)
    #print confusion matrix
    print('T\\P\tAlive\tDied\tSum\nAlive\t{}\t{}\t{}\nDied\t{}\t{}\t{}\nSum\t{}\t{}\t{}\n'.format(tn,fp,tn+fp,
                                                                                        fn,tp,fn+tp,
                                                                                        tn+fn,fp+tp,tn+fp+fn+tp))
    print(classification_report(result['SURVIVAL_STATUS'].values, result['predict_label'].values))
    print('AUC score:',roc_auc_score(result['SURVIVAL_STATUS'].values,result['score'].values))
    print('Overall accuracy:',accuracy_score(result['SURVIVAL_STATUS'].values, 
                                             result['predict_label'].values,
                                             normalize=True))