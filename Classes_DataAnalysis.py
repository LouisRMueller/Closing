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
sns.set_palette('Set1')

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
		pass
	# 	limit = 0.35
	# 	namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders", all_limit="all limit orders")
	# 	locdict = dict(bid_limit='lower center', ask_limit='upper center', all_limit='upper center')
	#
	# 	tmp = self._bcs_data.loc[mode, :].xs(stock, level='Symbol')
	# 	tmp = (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10000
	# 	tmp = tmp.unstack(level='Percent', fill_value=np.nan)
	# 	tmp = tmp.iloc[:, tmp.columns <= limit]
	#
	# 	tmp.plot(figsize=figsize, linewidth=1)
	# 	plt.hlines(0, 0, len(tmp.index), 'k', lw=1, linewidth=1)
	# 	plt.xlabel("")
	# 	plt.ylabel("Deviation from closing price in bps")
	# 	plt.title("{}: Gradual removal of {}".format(stock, namedict[mode]))
	#
	# 	plt.legend(loc=locdict[mode], ncol=int(len(tmp.columns)),
	# 			 labels=[str(int(x * 100)) + " \%" for x in tmp.columns])
	#
	# 	plt.tight_layout()
	# 	plt.savefig(figdir + "\\SensitivityRough\\remove_{}_{}".format(mode, stock), dpi=dpi)
	# 	plt.close()

	def plt_rmv_limit_aggregated(self):
		"""Only works with rough percentages"""
		limit = 0.35
		namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders", all_limit="all limit orders")
		locdict = dict(bid_limit='lower center', ask_limit='upper center', all_limit='upper center')

		for mode in iter(namedict.keys()):
			raw = copy.deepcopy(self._bcs_data.loc[mode, :])
			df = pd.DataFrame({'Deviation':(raw['adj_price'] - raw['close_price']) / raw['close_price'] * 10000})
			df.reset_index(inplace=True)
			df = df[df['Percent'] <= limit]
			df['Percent'] = (df['Percent'] * 100).astype(int).astype(str) + " \%"

			fig, ax1 = plt.subplots(1,1, figsize=figsize, dpi=dpi)
			sns.lineplot(x='Date', y='Deviation', hue='Percent', data=df, ax=ax1,
					   palette='gist_earth', lw=1, ci=95, n_boot=500)
			ax1.grid(which='major', axis='y')
			ax1.set_xlabel("")
			ax1.set_ylabel("Deviation from closing price in bps")
			ax1.set_title("Average impact of gradual removal of {} across SLI (Bootstrapped 95\% CI)".format(namedict[mode]))
			loca = dates.MonthLocator()
			form = dates.ConciseDateFormatter(loca)
			ax1.xaxis.set_major_locator(loca)
			ax1.xaxis.set_major_formatter(form)

			fig.tight_layout()
			plt.savefig(figdir + "\\SensitivityRough\\agg_remove_{}".format(mode))
			fig.show()
			plt.close()

		return raw

	def plt_rmv_limit_quant(self):
		limit = 0.45
		namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders",
					 all_limit="bid/ask limit orders")

		for mode in iter(namedict.keys()):
			raw = copy.deepcopy(self._bcs_data.loc[mode, :])
			quants = raw.groupby(['Date', 'Symbol']).first()
			quants = quants.groupby('Date')['close_price'].transform(lambda x: pd.qcut(x, 3, labels=range(1, 4)))
			quants.rename('quantile', inplace=True)
			quants.replace({1: 'least liquid', 2: 'neutral', 3: 'most liquid'}, inplace=True)
			tmp = raw[['close_price', 'adj_price']].join(quants, on=['Date', 'Symbol'], how='left')
			df = pd.DataFrame({'Deviation': (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10**4,
						    'Quantile': tmp['quantile']}).reset_index()
			df = df[df['Percent'] <= limit]
			df['Percent'] = (df['Percent'] * 100).astype(int).astype(str) + '\%'
			print(df.head())

			fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
			sns.boxplot(data=df, x='Percent', y='Deviation', palette='Set3',
					  ax=ax1, hue='Quantile', showfliers=False, linewidth=1)
			ax1.set_xlabel('')
			ax1.set_title("Quantile distribution of closing price deviations when removing {}".format(namedict[mode]))
			ax1.set_ylabel("Deviation in bps")
			ax1.set_xlabel("Amount of liquidity removed")
			ax1.grid(which='major', axis='y')
			fig.tight_layout()
			plt.savefig(figdir + "\\SensitivityRough\\Quantile_distribution_{}".format(mode))
			fig.show()
			plt.close()


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

			fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
			sns.lineplot(x='Percent', y='deviation', hue='log10(volume)',
					   data=cl, linewidth=0.75, palette='YlOrRd', ax=ax1)
			ax1.hlines(0, cl['Percent'].min(), cl['Percent'].max(), 'k', lw=1, linewidth=1.0)
			ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax1.set_xlabel("Removed liquidity")
			ax1.set_ylabel("Deviation from actual closing [bps]")
			ax1.set_title("Sensitivity of {} titles with respect to {} (N = {})"
					    .format(n, figdict[mode]['name'], str(len(numstocks[n]))))
			handles, labels = ax1.get_legend_handles_labels()
			ax1.grid(which='major', axis='both')
			ax1.legend(handles=handles[1:], labels=[round(float(l), 1) for l in labels[1:]],
					 loc=figdict[mode]['loc'], fontsize='small', title="log10(volume)")
			fig.tight_layout()
			plt.savefig(figdir + "\\SensitivityFine\\Sens_Percent_{}_{}".format(mode, n))
			# plt.show()
			plt.close()

	def plt_cont_rmv_agg(self):
		"""
		Plot only used on the fine measurements of the data.
		"""
		raw = copy.deepcopy(self._bcs_data.loc[['bid_limit', 'ask_limit', 'all_limit'], :])

		turn_df = pd.DataFrame({'turnover': raw['close_vol'] * raw['close_price'],
						    'adj_turnover': raw['adj_vol'] * raw['adj_price']})

		df = pd.DataFrame({'Deviation': (raw['adj_price'] - raw['close_price']) / raw['close_price'] * 10 ** 4,
					    'Volume': raw['adj_vol'],
					    'Volume Delta': (raw['adj_vol'] - raw['close_vol']) / raw['close_vol'],
					    'Turnover': turn_df['adj_turnover'] - turn_df['turnover'],
					    'Turnover Delta': (turn_df['adj_turnover'] - turn_df['turnover']) / turn_df['turnover']
					    }).reset_index()

		df.loc[df['Mode'] == 'all_limit', 'Percent'] = 2 * df.loc[df['Mode'] == 'all_limit', 'Percent'].values
		df.replace({'bid_limit': 'bid limit', 'ask_limit': 'ask limit', 'all_limit': 'all limit'},
				 inplace=True)

		print(df.head())
		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		xmin, xmax = 0, 0.5
		sns.lineplot(data=df, x='Percent', y='Deviation', hue='Mode',
				   ax=ax1, ci='sd', palette='Set2')
		ax1.grid(which='both')
		ax1.hlines(0,xmin,xmax, 'k', lw=1)
		ax1.set_xlim([xmin, xmax])
		ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		ax1.set_ylabel("Deviation in bps")
		ax1.set_xlabel("Removed percentage of limit orders")
		ax1.set_title("Average deviation of closing price depending on removed liquidity on SLI")
		plt.tight_layout()
		plt.savefig(figdir + "\\SensitivityFine\\Avg_closeprice_SLI.png")
		plt.show()
		plt.close()

		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(data=df, x='Percent', y='Volume Delta', hue='Mode',
				   ax=ax1, ci='sd', palette='Set2')
		ax1.set_xlim([xmin, xmax])
		ax1.hlines(0,xmin,xmax, 'k', lw=1)
		ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		ax1.grid(which='both')
		ax1.set_ylabel("Deviation in \%")
		ax1.set_xlabel("Removed percentage of limit orders")
		ax1.set_title("Average deviation of volume depending on removed liquidity on SLI")
		plt.tight_layout()
		plt.savefig(figdir + "\\SensitivityFine\\Avg_volume_dev_SLI.png")
		plt.show()
		plt.close()

		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(data=df, x='Percent', y='Turnover Delta', hue='Mode',
				   ax=ax1, ci='sd', palette='Set2')
		ax1.set_xlim([xmin, xmax])
		ax1.hlines(0,xmin,xmax, 'k', lw=1)
		ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		ax1.grid(which='both')
		ax1.set_ylabel("Deviation in \%")
		ax1.set_xlabel("Removed percentage of limit orders")
		ax1.set_title("Average deviation of turnover depending on removed liquidity on SLI")
		plt.tight_layout()
		plt.savefig(figdir + "\\SensitivityFine\\Avg_turnover_dev_SLI.png")
		plt.show()
		plt.close()

		return df


class DiscoAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._raw_data.set_index(['Date', 'Symbol'], inplace=True)
		self._raw_vol_classify()

	def plt_closing_volume(self):
		limit = 50

		meanvol = self._raw_data.loc[:, 'close_vol'].groupby('Symbol').mean()
		tmp_vol = meanvol / meanvol.max()
		tmp_vol.sort_values(ascending=False, inplace=True)
		tmp_vol = tmp_vol[tmp_vol > 0]

		# Volume contribution among all stocks without labels
		fig, ax = plt.subplots(1, 1, figsize=figsize)
		ax.bar(np.arange(1, len(tmp_vol) + 1), tmp_vol.values, width=1)
		ax.set_yscale('log')
		ax.set_ylabel("Volume divided by largest")
		ax.set_xlabel("Sorted titles")
		ax.set_title("Average daily closing volume normalized")
		plt.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\volume_percent_aggregated", dpi=dpi)
		plt.close()

		# Volume percentage of closings with labels
		fig, ax = plt.subplots(1, 1, figsize=figsize)
		ax.bar(tmp_vol.index[:limit], tmp_vol[:limit].values, width=0.7)
		ax.tick_params(axis='x', rotation=90)
		ax.set_title("Average daily closing volume normalized for {} most liquid titles".format(str(limit)))
		ax.set_ylabel("Volume divided by largest")
		plt.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\volume_percent_stocks", dpi=dpi)
		plt.close()

	def plt_deviation_discovery(self):
		raw = self._bcs_data

		quants = raw.groupby('Date')['close_vol'].transform(lambda x: pd.qcut(x, 3, labels=range(1, 4)))
		quants.rename('quantile', inplace=True)
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
			sns.lineplot(x='Date', y='dev', data=df[df['quantile'] == qt],
					   lw=1, palette='Set1', ax=ax, ci='sd')
			ax.hlines(0, xmin, xmax, 'k', lw=1)
			ax.set_ylim((-50, 50))
			ax.set_xlim((xmin, xmax))
			ax.set_ylabel("")
			ax.set_xlabel("")
			# ax.legend().remove()
			ax.set_title("Deviation of closing price to continuous midpoint for {} SLI titles (N = 10)".format(qt))
			ax.xaxis.set_major_locator(loca)
			ax.xaxis.set_major_formatter(form)

		axes[1].set_ylabel("Deviation [bps]")
		plt.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Closing_Deviations_terciles.png")
		plt.show()
		fig.clf()
		plt.close(fig)


		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(x='Date', y='dev', data=df, ax=ax1, lw=1, ci=95)
		ax1.hlines(0, xmin, xmax, 'k', lw=1)
		ax1.xaxis.set_major_locator(loca)
		ax1.xaxis.set_major_formatter(form)
		ax1.set_ylim((-60, 60))
		ax1.set_xlim((xmin, xmax))
		ax1.set_xlabel("")
		ax1.set_ylabel("Deviation [bps]")
		ax1.set_title("Average daily deviation of closing price from last midpoint over SLI (Bootstrapped 95\% CI)")
		plt.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Closing_Deviations_aggregated")
		plt.show()
		fig.clf()
		plt.close(fig)

		return df

	def plt_disco_distr_xsect(self, limit=15):
		# stocks = ['NESN','NOVN','ROG','UBSG','CSGN']
		stocks = self._avg_vol.index[:20]
		raw = self._raw_data[self._raw_data.index.get_level_values(level='Symbol').isin(stocks)]

		df = pd.DataFrame({'Dev': (raw['actual_close_price'] - raw['pre_midquote']) / raw['pre_midquote'] * 10**4,
					    'Vol': raw['close_vol']})
		df.reset_index(inplace=True)
		ylim = 230
		df = df[abs(df['Dev']) <= ylim]

		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.boxplot(x='Symbol', y='Dev', palette='Set3', data=df, ax=ax1, order=stocks)
		ax1.set_ylabel("Deviation from midquote in bps")
		ax1.yaxis.set_major_locator(ticker.MultipleLocator(100))
		ax1.grid(which='major', axis='y', color='k', lw=0.8)
		ax1.set_ylim([-ylim, ylim])
		ax1.set_xlabel('')
		plt.xticks(rotation=90)
		ax1.set_title("Distribution of deviations of closing from pre-close midquote of {} most \"closed\" stocks".format(limit))
		fig.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Distribution_xsection")
		fig.show()
		plt.close()

		return df



file_bcs = os.getcwd() + "\\Data\\bluechips.csv"

file_data = os.getcwd() + "\\Exports\\Sensitivity_rough_v1.csv"
Sens = SensAnalysis(file_data, file_bcs)
t = Sens.plt_rmv_limit_quant()

# file_data = os.getcwd() + "\\Exports\\Price_Discovery_v1.csv"
# Disco = DiscoAnalysis(file_data, file_bcs)
# t = Disco.plt_disco_distr_xsect(20)

