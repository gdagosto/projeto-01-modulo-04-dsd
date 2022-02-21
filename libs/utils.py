from sklearn.metrics import classification_report, confusion_matrix
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_slice
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cria_fig_ax(figsize=(16, 9), **kwargs):
    """
    Initialize fig and axs with some basic default values
    """
    fig, axs = plt.subplots(figsize=figsize, **kwargs)
    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    return fig, axs


def formata_grafico(grf, titulo='', legenda=[], xlabel='', ylabel='', titulo_legenda=''):
    """
    Given a matplotlib chart, allows for some customization while following some global formatting rules
    """
    if (titulo != ''):
        grf.set_title(titulo, loc='left', fontsize=20)

    if len(legenda) > 0:
        grf.legend(legenda, title=titulo_legenda, bbox_to_anchor=(1.02, 1), loc='upper left')
    # else:
    #     grf.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    if (xlabel != ''):
        grf.set_xlabel(xlabel)
    if (ylabel != ''):
        grf.set_ylabel(ylabel)

    grf.spines['top'].set_visible(False)
    grf.spines['right'].set_visible(False)

    return grf

def change_df_boolean(df, colName, bins=['No', 'Yes']):
    """
    Given a dataframe and a boolean column, change it to the bins passed
    """
    df.loc[:, colName] = np.where(df[colName] == 0, bins[0], bins[1])



def boolean_hist_plot(df,hue,labels,x,ax,title=''):
    """
    Plot a stacked histogram plot to analyze features vs boolean feature (set as hue)
    """
    sns.histplot(data=df, x=x, hue=hue,
                 stat="probability", multiple="fill", ax=ax)

    ax = formata_grafico(ax, title, labels,
                         ylabel='Percentual', xlabel=title)


def binarize(y_proba, threshold):
    """
    Given an array of probabilities and a threshold, returns a 0/1 array
    """
    return np.where(y_proba >= threshold, 1, 0)

def metric_over_thresholds(metric, y_true, y_proba, num_partition=101, optimal='max'):
    """
    Calculates a metric over several different threshold values
    metric must be able to received a named field y_true, y_pred
    
    If the optimal metric is that with maximum value, set optimal=max,
    otherwise set optimal=min.
    """

    assert optimal in ['max', 'min'], "Parameter optimal must be max or min"

    thresholds = np.linspace(0, 1, num=num_partition)
    metrics = np.array([metric(y_true, binarize(y_proba, t))
                        for t in thresholds])

    if optimal == 'max':
        optimal_thresh = thresholds[np.argmax(metrics)]
    else:
        optimal_thresh = thresholds[np.argmin(metrics)]

    return thresholds, metrics, optimal_thresh



def report_and_cm(y_true,y_pred):
    """
    Plots classification report and confusion matrix of a prediction result
    """
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(conf_mat=cm)
    plt.show()



def optuna_plots(study):
    """
    Given an optuna study, show some optuna.visualization plots
    """
    fig = plot_optimization_history(study)
    fig.show()
    fig = plot_parallel_coordinate(study)
    fig.show()
    fig = plot_param_importances(study)
    fig.show()
    fig = plot_slice(study)
    fig.show()
