import os
import sys
import colorsys
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from IPython.display import display, clear_output
from itertools import product
from scipy import stats
from wordcloud import WordCloud

warnings.filterwarnings('ignore', category=UserWarning)
os.makedirs('./result/plot/', exist_ok=True)

if os.name == 'posix':
    plt.rcParams['font.sans-serif'] = ['Songti SC']
else:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def get_color_gradient(index, total):
    hue = index / total
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    return f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})'


class my_plot:
    def __init__(self, plot_df, plot_name=None):
        self.plot_df = pd.DataFrame(plot_df)
        self.plot_name = plot_name or ['Title', 'Xlabel', 'Ylabel', 'save_name']

    def save_plot(self):
        plt.savefig(f'./result/plot/{self.plot_name[3]}.jpg', bbox_inches='tight', dpi=300)

    def line_plot(self, type=['-'], legend=True, ncol=0, font_scale=0.7,
                  x_label_num=6, save_bool=True, legend_loc='best',
                  rotation_angel=0, x_log=False, y_log=False, plt_show=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        fontsize = 12
        for col in self.plot_df.columns:
            ax.plot(self.plot_df.index, self.plot_df[col].values, type[0], label=col)
        ax.grid()
        ax.set_title(self.plot_name[0], fontsize=fontsize)
        ax.set_xlabel(self.plot_name[1], fontsize=fontsize)
        ax.set_ylabel(self.plot_name[2], fontsize=fontsize)
        if ncol > 0:
            plt.legend(fontsize=fontsize * font_scale, loc=legend_loc, ncol=ncol, bbox_to_anchor=(1, -0.1))
        else:
            plt.legend()
        self._set_log_axis(ax, x_log, y_log, x_label_num)
        plt.xticks(rotation=rotation_angel)
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)
        if save_bool:
            self.save_plot()
        if plt_show:
            plt.show()

    def line_go_area(self, rainbow_len=None):
        fig = go.Figure()
        for i, col in enumerate(self.plot_df.columns):
            kw = {}
            if rainbow_len is not None:
                kw['line'] = dict(color=get_color_gradient(i, rainbow_len))
            fig.add_trace(go.Scatter(x=list(self.plot_df.index), y=list(self.plot_df[col]), name=col, **kw))
        fig.update_layout(title_text=self.plot_name[0],
                          xaxis=dict(title_text=self.plot_name[1]),
                          yaxis=dict(title_text=self.plot_name[2]))
        fig.write_html(f'./result/plot/{self.plot_name[3]}.html')
        fig.show()

    def line_go_drag(self, is_date=False):
        data = [go.Scatter(x=self.plot_df.index, y=self.plot_df[col], name=col)
                for col in self.plot_df.columns]
        layout = go.Layout(xaxis=dict(
            rangeselector=dict(buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')])),
            rangeslider=dict(visible=True),
            type='date' if is_date else None))
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(title_text=self.plot_name[0])
        fig.update_xaxes(title_text=self.plot_name[1])
        fig.update_yaxes(title_text=self.plot_name[2])
        fig.write_html(f'./result/plot/{self.plot_name[3]}.html')
        fig.show()

    def hist_plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(self.plot_df, bins=20)
        ax.set_title(self.plot_name[0], fontsize=12)
        ax.set_xlabel(self.plot_name[1], fontsize=12)
        ax.set_ylabel(self.plot_name[2], fontsize=12)
        plt.savefig(f'./result/plot/{self.plot_name[3]}_hist_plot.jpg', bbox_inches='tight', dpi=300)

    def quantiles_plot(self):
        qtl_df = self.plot_df.dropna().rank() / len(self.plot_df.dropna())
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(qtl_df['x'], qtl_df['y'], s=0.00001)
        ax.set_title(self.plot_name[0], fontsize=12)
        ax.set_xlabel(self.plot_name[1], fontsize=12)
        ax.set_ylabel(self.plot_name[2], fontsize=12)
        ax.grid()
        plt.savefig(f'./result/plot/{self.plot_name[3]}_quantiles_plot.jpg', bbox_inches='tight', dpi=300)

    def scatter_plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(self.plot_df.index, self.plot_df.values, s=10)
        ax.set_title(self.plot_name[0], fontsize=12)
        ax.set_xlabel(self.plot_name[1], fontsize=12)
        ax.set_ylabel(self.plot_name[2], fontsize=12)
        ax.grid()
        plt.savefig(f'./result/plot/{self.plot_name[3]}_scatter_plot.jpg', bbox_inches='tight', dpi=300)

    def bar_plot(self, rotation_angel=0, width=0.3, plt_show=True):
        fig, ax = plt.subplots(figsize=(10, 3))
        for col in self.plot_df.columns:
            ax.bar(self.plot_df.index, self.plot_df[col].values, label=col, alpha=0.3, width=width)
        ax.set_title(self.plot_name[0], fontsize=12)
        ax.set_xlabel(self.plot_name[1], fontsize=12)
        ax.set_ylabel(self.plot_name[2], fontsize=12)
        plt.xticks(rotation=rotation_angel)
        plt.savefig(f'./result/plot/{self.plot_name[3]}_bar_plot.jpg', bbox_inches='tight', dpi=300)
        if plt_show:
            plt.show()

    def multi_hist_plot(self, nrows, ncols):
        figsize = (ncols * ncols + nrows, nrows * nrows + ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=300)
        for (r, c), k in zip(product(range(nrows), range(ncols)), range(nrows * ncols)):
            s = self.plot_df.iloc[:, k]
            axes[r, c].hist(s, density=True, bins=20)
            axes[r, c].set_title(f'{self.plot_df.columns[k]} 频率直方图', fontsize=12)
            mu, sigma = s.mean(), s.std()
            norm_x = np.sort(s.dropna())
            norm_y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((norm_x - mu) / sigma) ** 2)
            axes[r, c].plot(norm_x, norm_y, '--', color='tab:orange')
        plt.savefig(f'./result/plot/{self.plot_name[3]}_hist_plot.jpg', bbox_inches='tight', dpi=300)

    def multi_line_plot(self, nrows, ncols):
        figsize = (ncols * ncols + nrows, nrows * nrows + ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=300)
        for (r, c), k in zip(product(range(nrows), range(ncols)), range(nrows * ncols)):
            s = self.plot_df.iloc[:, k]
            mu, sigma = s.mean(), s.std()
            y = np.sort(s.values)
            y_rank = np.sort(s.rank(pct=True))
            x = np.sort(stats.norm.ppf(y_rank, mu, sigma))
            axes[r, c].plot(x, y, '.')
            axes[r, c].set_title(f'{self.plot_df.columns[k]} QQ图', fontsize=12)
            axes[r, c].grid()
            axes[r, c].plot(y, y, '--', color='tab:orange')
        plt.savefig(f'./result/plot/{self.plot_name[3]}_multi_line_plot.jpg', bbox_inches='tight', dpi=300)

    def price_volume_create(self, plot_name='Ylabel2'):
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(4, 1)
        ax1 = fig.add_subplot(gs[:3, :])
        ax2 = fig.add_subplot(gs[3, :])
        ax1.set_title(self.plot_name[0], fontsize=12)
        for col in self.plot_df.columns:
            ax1.plot(self.plot_df.index, self.plot_df[col].values, label=col)
        ax1.legend()
        ax1.set_ylabel(self.plot_name[2], fontsize=12)
        ax2.set_xlabel(self.plot_name[1], fontsize=12)
        ax2.set_ylabel(plot_name, fontsize=12)
        sys.stdout = open(os.devnull, 'w')
        return ax1, ax2

    def price_volume_plot(self, ax1, ax2, plot_name='Ylabel2', width=0.4,
                          x_log=False, y_log1=False, y_log2=False,
                          rotation_angle=0, ax2_type='bar', x_label_num=6, plt_save=True):
        color_set = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fontsize = 12
        plot_df1 = self.plot_df.copy()
        plot_df1.iloc[:, -1] = np.nan
        for i, col in enumerate(plot_df1.columns):
            ax1.plot(plot_df1.index, plot_df1[col].values, label=col, color=color_set[i])
        self._set_log_axis(ax1, x_log, y_log1, x_label_num)
        ax1.grid(True)
        ax1.set_xticklabels([''] * len(ax1.get_xticks()))

        if ax2_type == 'linear':
            plot_df2 = self.plot_df.copy()
            plot_df2.iloc[:, :-1] = np.nan
            for i, col in enumerate(plot_df2.columns):
                ax2.plot(plot_df2.index, plot_df2[col].values, label=col, color=color_set[i])
            ax2.grid(True)
        elif ax2_type == 'bar':
            ax2.bar(self.plot_df.index, self.plot_df.iloc[:, -1].values,
                    label=self.plot_df.columns[-1], alpha=0.3, width=width)
        self._set_log_axis(ax2, x_log, y_log2, x_label_num)
        ax2.set_xlim(ax1.get_xlim())
        plt.xticks(rotation=rotation_angle)
        plt.tight_layout()
        if plt_save:
            plt.savefig(f'./result/plot/{self.plot_name[3]}_bar_plot.jpg', bbox_inches='tight', dpi=300)

    def heat_map(self):
        fig_w = 6.2 * len(self.plot_df) / 8
        fig_h = 5 * len(self.plot_df) / 8
        fig, axes = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        sns.heatmap(self.plot_df.sort_index(ascending=False), annot=True, cmap='plasma_r', ax=axes)
        axes.set_title(self.plot_name[0], fontsize=11)
        axes.set_xlabel(self.plot_name[1], fontsize=11)
        axes.set_ylabel(self.plot_name[2], fontsize=11)
        plt.savefig(f'./result/plot/{self.plot_name[3]}.png')

    def word_cloud_plot(self):
        self.plot_df.columns = ['words', 'sizes']
        word_sizes = dict(zip(self.plot_df['words'], self.plot_df['sizes']))
        wc = WordCloud(width=800, height=400, background_color='white', max_words=50,
                       random_state=230412, relative_scaling=0.5, colormap='viridis').generate_from_frequencies(word_sizes)
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud of {self.plot_name[0]}', fontsize=30)
        plt.tight_layout(pad=0)
        plt.savefig(f'./result/{self.plot_name[3]}_word_plot.jpg', bbox_inches='tight', dpi=300)
        plt.show()

    def _set_log_axis(self, ax, x_log, y_log, x_label_num):
        if x_log:
            step_size = len(self.plot_df.index) // x_label_num
            selected_index = list(self.plot_df.index[::step_size])
            ax.set_xscale('log')
            ax.set_xticks(selected_index)
            ax.set_xticklabels([f'{i:.2e}' for i in ax.get_xticks()])
        if y_log:
            ax.set_yscale('log')
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([f'{i:.2e}' for i in ax.get_yticks()])
