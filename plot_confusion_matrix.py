from __future__ import division
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def whatClassesAreResponsibleForEachClassification(confusionMatrix):
    #print np.sum(confusionMatrix, axis=0)
    return confusionMatrix / np.sum(confusionMatrix, axis=0)

def howEachClassHasBeenClassified(confusionMatrix):
    return (confusionMatrix.T / np.sum(confusionMatrix, axis=1)).T

def getInvertedNormalizedConfusionMatrix(realTargets, predictions):
    return whatClassesAreResponsibleForEachClassification(
        confusion_matrix(y_true=realTargets, y_pred=predictions)
    )

def getNormalizedConfusionMatrix(realTargets, predictions):
    return howEachClassHasBeenClassified(
        confusion_matrix(y_true=realTargets, y_pred=predictions)
    )

def plot_confusion_matrix(y_true, y_pred, normalized=False, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix.
    Plot confusion matrix by using seaborn heatmap function
    If normalized is set to True, the rows of the confusion matrix are normalized so that they sum up to 1.
    
    """
    cm = getNormalizedConfusionMatrix(y_true, y_pred) if normalized else confusion_matrix(y_true=y_true, y_pred=y_pred)
    #print cm

    vmin, vmax = (0., 1.) if normalized is True else (None, None)

    classes = np.unique(y_true) if classes is None else classes

    fmt = '.2f' if normalized else 'd'

    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=vmin, vmax=vmax, 
                annot=True, annot_kws={"fontsize":9},
                fmt=fmt,
                )

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

