import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

def preprocess_data(stock, period):
    df = yf.download(tickers=[stock], period=f'{period}y')
    y = df.fillna(method='ffill')

    df.to_csv(f'./data/{stock}.csv')
    return y


def plot_data(df, main_title = '', save_path = None):
    df_plot = df.copy()

    ncols = 2
    nrows = int(round(df_plot.shape[1] / ncols))

    fig, ax = plt.subplots(nrows = nrows, ncols = ncols,
                            sharex = True, figsize = (14, 7))

    if main_title:
        fig.suptitle(main_title, fontsize = 16)

    for i, ax in enumerate(fig.axes):
        if i < df_plot.shape[1]:
            sns.lineplot(data = df_plot.iloc[:, i], ax = ax)
            ax.set_title(df_plot.columns[i])

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches = 'tight')
    else:
        plt.show()



if __name__ == "__main__":
    data = preprocess_data('./GOOG.csv')
    plot_data(data, '../Images/GOOG.png')