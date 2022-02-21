import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

def title_and_labels(ax, title, xlabel, ylabel, fontsize, color='black', rotation=0):
    ax.set_title(title, fontsize=fontsize, fontweight='bold', color=color)
    ax.set_xlabel(xlabel, fontsize=fontsize-4, fontweight='bold', color=color)
    ax.set_ylabel(ylabel, fontsize=fontsize-4, fontweight='bold', color=color)
    ax.tick_params(axis='both', colors=color)
    plt.tick_params(axis='x', rotation=rotation)


def annotate_bar(ax, mode='abs'):
    for p in ax.patches:
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        value = p.get_height()

        if mode == 'abs':
            if value > 10000:
                text = str(int(value/1000))+'k'
            else:
                text = int(value)
        else:
            text = str(round(value*100, 1))+'%'
        ax.annotate(
            text,
            (x, y),
            ha='center', va='bottom',
            fontsize=12, color='#337',
            # [ 'normal' | 'bold' | 'heavy' | 'light' | 'ultrabold' | 'ultralight']
            fontweight='bold',
        )


def plot_chances(df, x, y, tick_dict, pallete, hue=None, ax=None, figsize=(7, 5)):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    df = (df[[x, y]].groupby(x).agg('mean'))

    sns.barplot(data=df.reset_index(), x=x, y=y,
                palette=pallete, ax=ax, hue=hue)

    # title_and_labels(
    #     ax=ax,
    #     title=title,
    #     xlabel=xlabel,
    #     ylabel=ylabel,
    #     # fontsize=18,
    # )

    ticks = ax.get_xticks()
    ax.set_xticklabels(list(map(tick_dict.get, ticks)))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def desenha_seta(ax):
    '''
        Dado um ax de um grafico de barras binário, 
        desenha uma seta que representa o aumento de uma barra para a outra
    '''
    xy = []  # coordenadas do centro do topo de cada barra
    for p in ax.patches:
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        xy.append((x, y))

    #desenha as setas
    ax.annotate("",
                xy=xy[1], xycoords='data',  # a seta chega na coordenada xy[0]
                # a seta sai da coordenada xy[1]
                xytext=xy[0], textcoords='data',
                arrowprops=dict(arrowstyle="simple", color='#B88',
                                connectionstyle="angle3,angleA=10,angleB=90",  # angulos de chegada e saída
                                ),
                )
    #quantidade de vezes que uma barra é maior que a outra
    text = str(round(np.array(xy)[1, 1]/np.array(xy)[0, 1], 1))+'x'

    ax.text(*np.array(xy).mean(axis=0),  # desempacota as coordenadas (de tupla para 2 valores)
            text, weight="bold", fontsize=25, color='#722')
