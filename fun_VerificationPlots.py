import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import dates
from matplotlib import ticker
from matplotlib.colors import LogNorm
import seaborn as sns
import copy

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# register_matplotlib_converters()

conv = 2.54
dpi = 300
figsize = (22 / conv, 13 / conv)
figdir = os.getcwd() + "\\01 Presentation December\\Figures"

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
def_palette = "Set1"
sns.set_palette(def_palette, 2)
def_color = sns.color_palette(def_palette, 1)[0]


def plot_closing_orders(dump, stock, date='2019-03-15'):
	dic = dump[date][stock]

	for p in [0]:
		df = dic[p]
		opt_price = abs(df['OIB']).idxmin()
		print(opt_price)
		df = df[['cumulative bids', 'cumulative asks']].stack()

		df = df.reset_index(drop=False)
		print(df.head())
		df.columns = ['price', 'side','shares']
		df.sort_values('price', ascending=True, inplace=True)

		xmin, xmax = df['price'].min(), opt_price * 2

		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(ax=ax, data=df, x='price', y='shares', hue='side', markers='.')
		l1, l2 = ax.lines[0], ax.lines[1]
		x1, y1 = l1.get_xydata()[:,0], l1.get_xydata()[:,1]
		x2, y2 = l2.get_xydata()[:,0], l2.get_xydata()[:,1]
		ax.fill_between(x1,y1, y2, where= y1 > y2, alpha=0.3)
		ax.fill_between(x2,y1, y2, where= y1 < y2, alpha=0.3)
		ax.set_title("Cumulative Bid/Ask by price available at closing auction for {} on {}".format(stock, date))
		ax.set_xlim(left=xmin, right=xmax)
		ax.set_ylim(bottom=0)
		ax.axvline(opt_price, color='k', lw=1, ls='dashed')
		ax.axhline()

		ax.xaxis.set_major_locator(ticker.MaxNLocator(10))

		plt.show()
		plt.close()

plot_closing_orders(dump, 'UBSG')
