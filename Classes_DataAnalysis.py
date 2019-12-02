import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import dates
from matplotlib import ticker
import seaborn as sns
import copy

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

register_matplotlib_converters()

conv = 2.54
dpi = 400
figsize = (22 / conv, 13 / conv)
figdir = os.getcwd() + "\\01 Presentation December\\Figures"

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)


class DataAnalysis:
	def __init__(self, datapath, bluechippath):
		self._bluechips = pd.read_csv(bluechippath)['symbol']
		self._raw_data = pd.read_csv(datapath, parse_dates=['Date'])
		self._figpath = os.getcwd() + "\\01 Presentation December"
		print("--- Class Initiated ---")
	
	def _raw_vol_classify(self):
		self._bcs_data = self._raw_data[self._raw_data.index.get_level_values(level='Symbol').isin(self._bluechips)]
		self._avg_vol = self._raw_data['close_vol'].groupby('Symbol').mean().sort_values(ascending=False).dropna()


class SensAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._raw_data['Percent'] = self._raw_data['Percent'].round(2)
		self._raw_data.set_index(['Mode', 'Date', 'Symbol', 'Percent'], inplace=True)
		self._raw_vol_classify()
	
	def plt_remove_limit_individual(self, stock, mode):
		limit = 0.35
		namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders", all_limit="all limit orders")
		locdict = dict(bid_limit='lower center', ask_limit='upper center', all_limit='upper center')
		
		tmp = self._bcs_data.loc[mode, :].xs(stock, level='Symbol')
		tmp = (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10000
		tmp = tmp.unstack(level='Percent', fill_value=np.nan)
		tmp = tmp.iloc[:, tmp.columns <= limit]
		
		tmp.plot(figsize=figsize, linewidth=1)
		plt.hlines(0, 0, len(tmp.index), 'k', 'dashed', linewidth=1)
		plt.xlabel("")
		plt.ylabel("Deviation from closing price in bps")
		plt.title("{}: Gradual removal of {}".format(stock, namedict[mode]))
		
		plt.legend(loc=locdict[mode], ncol=int(len(tmp.columns)),
		           labels=[str(int(x * 100)) + " \%" for x in tmp.columns])
		
		plt.tight_layout()
		plt.savefig(figdir + "\\SensitivityRough\\remove_{}_{}".format(mode, stock), dpi=dpi)
		plt.close()
	
	def plt_rmv_limit_aggregated(self, mode, aggreg='mean'):
		limit = 0.35
		namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders", all_limit="all limit orders")
		aggdict = dict(mean='Average', median='Median')
		locdict = dict(bid_limit='lower center', ask_limit='upper center', all_limit='upper center')
		tmp = copy.deepcopy(self._bcs_data.loc[mode, :])
		
		if aggreg == 'mean':
			tmp = tmp.groupby(['Date', 'Percent']).mean()
		elif aggreg == 'median':
			tmp = self._bcs_data.loc[mode, :].groupby(['Date', 'Percent']).median()
		
		tmp = (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10000
		tmp = tmp.unstack(level='Percent', fill_value=np.nan)
		tmp = tmp.loc[:, tmp.columns <= limit]
		
		fig, ax1 = plt.subplots(figsize=figsize)
		tmp.plot(figsize=figsize, linewidth=1, ax=ax1)
		ax1.hlines(0, tmp.index[0], tmp.index[-1], 'k', 'dashed', linewidth=1)
		ax1.set_xlabel("")
		ax1.set_ylabel("Deviation from closing price in bps")
		ax1.set_title("{} impact of gradual removal of {} across SLI".format(aggdict[aggreg], namedict[mode]))
		
		ax1.legend(loc=locdict[mode], ncol=int(len(tmp.columns)),
		           labels=[str(int(x * 100)) + " \%" for x in tmp.columns])
		
		loca = dates.MonthLocator()
		form = dates.ConciseDateFormatter(loca)
		ax1.xaxis.set_major_locator(loca)
		ax1.xaxis.set_major_formatter(form)
		
		plt.tight_layout()
		plt.savefig(figdir + "\\SensitivityRough\\agg_remove_{}_{}".format(mode, aggreg), dpi=dpi)
		plt.close()
		
		return tmp
	
	def plt_rmv_limit_quant(self, mode):
		limit = 0.35
		namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders",
		                all_limit="bid/ask limit orders")
		raw = copy.deepcopy(self._bcs_data.loc[mode, :])
		
		quants = raw.groupby(['Date', 'Symbol']).first()
		quants = quants.groupby('Date')['close_price'].transform(lambda x: pd.qcut(x, 3, labels=range(1, 4)))
		quants.rename('quantile', inplace=True)
		tmp = raw[['close_price', 'adj_price']].join(quants, on=['Date', 'Symbol'], how='left')
		
		tmp = tmp.groupby(['quantile', 'Date', 'Percent']).mean()
		tmp = (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10000
		tmp = tmp.unstack(level='Percent', fill_value=np.nan)
		tmp = tmp.loc[:, [round(x * 0.1, 1) for x in range(1, 5)]]
		
		fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
		for ax, qt in zip(axes, range(1, 4)):
			plot_df = tmp.xs(qt, level='quantile')
			xmin, xmax = plot_df.index[0], plot_df.index[-1]
			
			ax.plot(plot_df, linewidth=0.8)
			ax.set_xlim((xmin, xmax))
			ax.set_title("Quantile " + str(qt) + ": Deviation [bps] of closing price by removing " + namedict[mode])
			ax.hlines(0, xmin, xmax, 'k', 'dashed', linewidth=1.2)
		loca = dates.MonthLocator()
		format = dates.ConciseDateFormatter(loca)
		axes[-1].xaxis.set_major_locator(loca)
		axes[-1].xaxis.set_major_formatter(format)
		axes[-1].legend(labels=[str(int(x * 100)) + '\%' for x in plot_df.columns], loc='upper center',
		                bbox_to_anchor=[0.5, -0.3], ncol=len(plot_df.columns))
		plt.tight_layout()
		plt.savefig(figdir + "\\SensitivityRough\\quantile_remove_{}.png".format(mode), dpi=dpi)
		fig.clf()
		plt.close(fig)
	
	def plt_cont_rmv_indiv(self, mode):
		raw = copy.deepcopy(self._raw_data.loc[mode, :])
		raw = raw[raw['close_vol'] > 1000]
		# raw = copy.deepcopy(self._bcs_data.loc[mode, :])
		numstocks = {'available': self._avg_vol.index[self._avg_vol > 0],
		             'top 75': self._avg_vol.index[:75],
		             'SLI': self._bluechips,
		             'top 20': self._avg_vol.index[:20],
		             'top 10': self._avg_vol.index[:10]}
		figdict = dict(bid_limit=dict(name='bid limit orders', loc='lower left'),
		               ask_limit=dict(name='ask limit orders', loc='upper left'),
		               all_limit=dict(name='bid/ask limit orders', loc='upper left'))
		
		for n in numstocks.keys():
			cl = pd.DataFrame({'deviation': (raw['adj_price'] - raw['close_price']) / raw['close_price'] * 10 ** 4,
			                   'log10(volume)': np.log10(raw['close_vol'])})
			cl = cl[cl.index.get_level_values('Symbol').isin(numstocks[n])]
			
			cl = cl.groupby(['Symbol', 'Percent']).mean()
			cl.reset_index(drop=False, inplace=True)
			
			fig, ax1 = plt.subplots(1, 1, figsize=figsize)
			sns.lineplot(x='Percent', y='deviation', hue='log10(volume)',
			             data=cl, linewidth=0.75, palette='coolwarm', ax=ax1)
			ax1.hlines(0, cl['Percent'].min(), cl['Percent'].max(), 'k', 'dashed', linewidth=1.0)
			ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax1.set_xlabel("Removed liquidity")
			ax1.set_ylabel("Deviation from actual closing [bps]")
			ax1.set_title("Sensitivity of {} titles with respect to {} (N = {})"
			              .format(n, figdict[mode]['name'], str(len(numstocks[n]))))
			handles, labels = ax1.get_legend_handles_labels()
			ax1.legend(handles=handles[1:], labels=[round(float(l), 1) for l in labels[1:]],
			           loc=figdict[mode]['loc'], fontsize='small', title="log10(volume)")
			fig.tight_layout()
			plt.savefig(figdir + "\\SensitivityFine\\Sens_Percent_{}_{}".format(mode, n))
			# plt.show()
			plt.close()
	
	def plt_cont_rmv_agg(self):
		raw = copy.deepcopy(self._bcs_data.drop(index=['all_market']))
		
		# modes = raw.index.get_level_values(level='Mode').unique()
		dev = (raw.loc[:, 'adj_price'] - raw.loc[:, 'close_price']) / raw.loc[:, 'close_price'] * 1000
		df = pd.DataFrame({'Deviation': dev, 'Volume': raw.loc[:, 'adj_vol']}).reset_index()
		
		df.replace({'bid_limit': 'bid limit', 'ask_limit': 'ask limit', 'all_limit': 'all limit'},
		           inplace=True)
		
		print(df.head())
		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		xmin, xmax = 0, df['Percent'].max()
		sns.lineplot(data=df, x='Percent', y='Deviation', hue='Mode', ax=ax1, ci=95, palette='Set2')
		ax1.hlines(0, xmin, xmax, 'k', 'dashed')
		ax1.set_xlim([xmin, xmax])
		ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		ax1.set_ylabel("Deviation in bps")
		ax1.set_xlabel("Removed percentage of limit orders")
		ax1.set_title("Average deviation of closing price depending on removed liquidity on SLI (95\% CI)")
		plt.tight_layout()
		plt.savefig(figdir + "\\SensitivityFine\\Avg_closeprice_SLI.png")
		plt.show()
		plt.close()
		
		return df


class DiscoAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._raw_data.set_index(['Date', 'Symbol'], inplace=True)
		self._raw_vol_classify()
	
	def plot_closings_per_stock(self):
		limit = 50
		
		meanvol = self._raw_data.loc[:, 'close_vol'].groupby('Symbol').mean()
		tmp_vol = meanvol / meanvol.max()
		tmp_vol.sort_values(ascending=False, inplace=True)
		tmp_vol = tmp_vol[tmp_vol > 0]
		
		# # Density Function of Closing frequencies
		# tmp_df = self._raw_data.loc[:, 'close_price_calculated'].groupby('Symbol').count()
		# numdays = self._raw_data.index.get_level_values(level='Date').nunique()
		# tmp_norm = tmp_df / numdays * 100
		# tmp_norm = tmp_norm[tmp_vol.index]
		# fig, ax = plt.subplots(1, 1, figsize=figsize)
		# ax.hist(tmp_norm.values, bins=100, density=True)
		# plt.show()
		
		# Volume contribution among all stocks without labels
		fig, ax = plt.subplots(1, 1, figsize=figsize)
		ax.bar(np.arange(1, len(tmp_vol) + 1), tmp_vol.values, width=1)
		ax.set_yscale('log')
		ax.set_ylabel("Volume divided by largest")
		ax.set_xlabel("Sorted titles")
		ax.set_title("Average daily closing volume normalized")
		plt.savefig(figdir + "\\PriceDiscovery\\volume_percent_aggregated", dpi=dpi)
		plt.close()
		
		# Volume percentage of closings with labels
		fig, ax = plt.subplots(1, 1, figsize=figsize)
		ax.bar(tmp_vol.index[:limit], tmp_vol[:limit].values, width=0.7)
		ax.tick_params(axis='x', rotation=90)
		ax.set_title("Average daily closing volume normalized for {} most liquid titles".format(str(limit)))
		ax.set_ylabel("Volume divided by largest")
		plt.savefig(figdir + "\\PriceDiscovery\\volume_percent_stocks", dpi=dpi)
		plt.close()
	
	def plt_deviation_discovery(self):
		raw = self._bcs_data
		
		quants = raw.groupby('Date')['close_vol'].transform(lambda x: pd.qcut(x, 3, labels=range(1, 4)))
		quants.rename('quantile', inplace=True)
		# quants = quants.astype(int)
		quants.replace({1: 'least liquid', 2: 'neutral', 3: 'most liquid'}, inplace=True)
		df = pd.DataFrame(dict(dev=(raw['actual_close_price'] - raw['pre_midquote']) / raw['pre_midquote'] * 10000,
		                       vol=raw['close_vol']))
		df = df.join(quants)
		df = df.reset_index()
		
		fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, dpi=dpi)
		xmin, xmax = df['Date'].iloc[0], df['Date'].iloc[-1]
		loca = dates.MonthLocator()
		form = dates.ConciseDateFormatter(loca)
		
		for ax, qt in zip(axes, df['quantile'].unique()):
			sns.lineplot(x='Date', y='dev', hue='quantile', data=df[df['quantile'] == qt],
			             lw=1, palette='Set1', ax=ax, ci=95)
			ax.hlines(0, xmin, xmax, 'k', 'dashed')
			ax.set_ylim((-50, 50))
			ax.set_xlim((xmin, xmax))
			ax.set_ylabel("")
			ax.set_xlabel("")
			ax.legend().remove()
			ax.set_title("Deviation of closing price to continuous midpoint for {} SLI titles (N = 10)".format(qt))
			ax.xaxis.set_major_locator(loca)
			ax.xaxis.set_major_formatter(form)
		
		axes[1].set_ylabel("Deviation [bps]")
		plt.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Closing_Deviations_terciles.png")
		# plt.show()
		plt.close()
		
		# df = df[df['Symbol'].isin(['Nesn','NOVN','ROG','UBSG','CSGN'])]
		
		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(x='Date', y='dev', data=df, ax=ax1, lw=1, palette='Set1', ci=95)
		ax1.hlines(0, xmin, xmax, 'k', 'dashed', lw=1)
		ax1.xaxis.set_major_locator(loca)
		ax1.xaxis.set_major_formatter(form)
		ax1.set_ylim((-50, 50))
		ax1.set_xlim((xmin, xmax))
		ax1.set_xlabel("")
		ax1.set_ylabel("Deviation [bps]")
		ax1.set_title("Average daily deviation of closing price from last midpoint over SLI (95\% CI)")
		plt.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Closing_Deviations_aggregated")
		# plt.show()
		plt.close(fig)


file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
file_data = os.getcwd() + "\\Exports\\Sensitivity_fine_v1.csv"
Sens = SensAnalysis(file_data, file_bcs)
t = Sens.plt_cont_rmv_agg()
#
# file_data = os.getcwd() + "\\Exports\\Price_Discovery_v1.csv"
# Disco = DiscoAnalysis(file_data, file_bcs)
# Disco.plt_deviation_discovery()
