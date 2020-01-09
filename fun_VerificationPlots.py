import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import dates
from matplotlib import ticker
from matplotlib.colors import LogNorm
import seaborn as sns
import itertools

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# register_matplotlib_converters()

conv = 2.54
dpi = 300
figsize = (22 / conv, 13 / conv)
figdir = os.getcwd() + "\\02 Slides January\\Figures"

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
def_palette = "Set1"
sns.set_palette(def_palette, 2)
def_color = sns.color_palette(def_palette, 1)[0]



def plot_closing_orders(dump, stock, mode, date='2019-03-15'):
	dic = dump[date][stock]
	fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi, sharey=True)
	baseprice = abs(dic[0.1]['OIB']).idxmin()
	xmin, xmax = baseprice * 0.98, baseprice * 1.02

	for p, a in zip(dic.keys(), itertools.product(range(2), range(3))):
		ax = axes[a[0], a[1]]
		df = dic[p]
		opt_price = abs(df['OIB']).idxmin()
		trade_vol = df['vol'].mean()
		df = df[['cum. bids', 'cum. asks']].stack()

		df = df.reset_index(drop=False)
		df.columns = ['price', 'side','shares']
		df.sort_values('price', ascending=True, inplace=True)

		sns.lineplot(ax=ax, data=df, x='price', y='shares', hue='side', markers='.', lw=1)
		l1, l2 = ax.lines[0], ax.lines[1]
		x1, y1 = l1.get_xydata()[:,0], l1.get_xydata()[:,1]
		x2, y2 = l2.get_xydata()[:,0], l2.get_xydata()[:,1]
		ax.fill_between(x1,y1, y2, where= y1 > y2, alpha=0.15)
		ax.fill_between(x2,y1, y2, where= y1 < y2, alpha=0.15)
		ax.set_title("{2}: {0: 0.0f}\% ({1})".format(-p*100, mode, stock), fontsize='medium')
		ax.set_xlim(left=xmin, right=xmax)
		# ax.set_ylim(bottom=0)
		ax.set_xlabel('calculated closing price: {0: 0.3f} CHF'.format(opt_price), fontsize='small')
		ax.set_ylabel('Number of shares')
		ax.axvline(opt_price, color='k', lw=0.8, ls='solid')
		ax.axhline(trade_vol, color='k', lw=1, ls='dotted')
		ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
		ax.legend(loc='lower left', fontsize='x-small')

	fig.tight_layout()
	# plt.savefig(figdir + "\\{}_{}_removal.png".format(stock, mode))
	plt.show()
	plt.close()