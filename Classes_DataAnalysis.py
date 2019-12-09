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

register_matplotlib_converters()

conv = 2.54
dpi = 400
figsize = (22 / conv, 13 / conv)
figdir = os.getcwd() + "\\01 Presentation December\\Figures"

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
def_color = "Set1"
sns.set_palette(def_color)


class DataAnalysis:
	def __init__(self, datapath, bluechippath):
		self._bluechips = pd.read_csv(bluechippath)['symbol']
		self._raw_data = pd.read_csv(datapath, parse_dates=['Date'])
		print("--- Class Initiated ---")

	def _raw_vol_classify(self):
		self._bcs_data = self._raw_data[self._raw_data.index.get_level_values(level='Symbol').isin(self._bluechips)]
		self._avg_vol = self._raw_data['close_vol'].groupby('Symbol').mean().sort_values(ascending=False).dropna()
		self._avg_turnover = self._raw_data['close_turnover'].groupby('Symbol').mean().sort_values(
			ascending=False).dropna()


class SensAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._raw_data['close_turnover'] = self._raw_data['close_price'] * self._raw_data['close_vol']
		self._raw_data['Percent'] = self._raw_data['Percent'].round(2)
		self._raw_data.set_index(['Mode', 'Date', 'Symbol', 'Percent'], inplace=True)
		self._raw_vol_classify()

	def plt_remove_limit_individual(self, stock, mode):
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
		# 	fig.tight_layout()
		# 	plt.savefig(figdir + "\\SensitivityRough\\remove_{}_{}".format(mode, stock), dpi=dpi)
		# 	plt.close()
		pass

	def plt_rmv_limit_aggregated(self):
		"""Only works with rough percentages"""
		limit = 0.2
		namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders", all_limit="all limit orders")
		locdict = dict(bid_limit='lower center', ask_limit='upper center', all_limit='upper center')

		for mode in iter(namedict.keys()):
			raw = self._bcs_data.loc[mode, :]
			df = pd.DataFrame({'Deviation': (raw['adj_price'] - raw['close_price']) / raw['close_price'] * 10000})
			df.reset_index(inplace=True)
			df = df[df['Percent'] <= limit]
			df['Percent'] = (df['Percent'] * 100).astype(int).astype(str) + " \%"

			fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
			sns.lineplot(x='Date', y='Deviation', hue='Percent', data=df, ax=ax1,
					   palette='gist_heat', lw=1, ci=95, n_boot=500)
			ax1.set_axisbelow(True)
			ax1.grid(which='major', axis='y')
			ax1.set_xlabel("")
			ax1.set_ylabel("Deviation from closing price in bps")
			ax1.set_title(
				"Average impact of gradual removal of {} across SLI (Bootstrapped 95\% CI)".format(namedict[mode]))
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
		limit = 0.2
		namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders",
					 all_limit="bid/ask limit orders")

		for mode in iter(namedict.keys()):
			raw = copy.deepcopy(self._bcs_data.loc[mode, :])
			quants = raw.groupby(['Date', 'Symbol']).first()
			quants = quants.groupby('Date')['close_turnover'].transform(lambda x: pd.qcut(x, 3, labels=range(1, 4)))
			quants.rename('quantile', inplace=True)
			quants.replace({1: 'least liquid', 2: 'neutral', 3: 'most liquid'}, inplace=True)

			tmp = raw[['close_price', 'adj_price', 'close_vol', 'adj_vol']].join(quants, on=['Date', 'Symbol'],
																    how='left')
			df = pd.DataFrame(
				{'Price Deviation': (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10 ** 4,
				 'Volume Deviation': (tmp['adj_vol'] - tmp['close_vol']) / tmp['close_vol'],
				 'Quantile': tmp['quantile']}).reset_index()
			df = df[df['Percent'] <= limit]
			df['Percent'] = (df['Percent'] * 100).astype(int).astype(str) + '\%'

			fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
			sns.boxplot(data=df, x='Percent', y='Price Deviation', palette='Set3', whis=[2.5, 97.5],
					  ax=ax1, hue='Quantile', showfliers=False, linewidth=1)
			ax1.set_xlabel('')
			ax1.set_title(
				"Quantile distribution of closing price deviations when removing {}".format(namedict[mode]))
			ax1.set_ylabel("Deviation in bps")
			ax1.set_xlabel("Amount of liquidity removed")
			ax1.grid(which='major', axis='y')
			fig.tight_layout()
			plt.savefig(figdir + "\\SensitivityRough\\Quantile_distribution_{}".format(mode))
			fig.show()
			plt.close()

			fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
			sns.boxplot(data=df, x='Percent', y='Volume Deviation', palette='Set3', whis=[2.5, 97.5],
					  ax=ax1, hue='Quantile', showfliers=False, linewidth=1)
			ax1.set_xlabel('')
			ax1.set_title(
				"Quantile distribution of closing volume deviations when removing {}".format(namedict[mode]))
			ax1.set_ylabel("Deviation in \%")
			ax1.set_xlabel("Amount of liquidity removed")
			ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
			ax1.grid(which='major', axis='y')
			fig.tight_layout()
			plt.savefig(figdir + "\\SensitivityRough\\Quantile_distribution_volume_{}".format(mode))
			fig.show()
			plt.close()

	def plt_cont_rmv_indiv(self, mode):
		limit = 0.2
		raw = copy.deepcopy(self._raw_data.loc[mode, :])
		raw = raw[raw['close_vol'] > 1000]
		numstocks = {'available': self._avg_turnover.index[self._avg_turnover > 0],
				   'top 120': self._avg_turnover.index[:120],
				   'top 60': self._avg_turnover.index[:60],
				   'SLI': self._bluechips,
				   'top 20': self._avg_turnover.index[:20]}
		figdict = dict(bid_limit=dict(name='bid limit orders', loc='lower left'),
					ask_limit=dict(name='ask limit orders', loc='upper left'),
					all_limit=dict(name='bid/ask limit orders', loc='upper left'))

		for n in numstocks.keys():
			cl = pd.DataFrame(
				{'Price Deviation': (raw['adj_price'] - raw['close_price']) / raw['close_price'] * 10 ** 4,
				 'Volume Deviation': (raw['adj_vol'] - raw['close_vol']) / raw['close_vol'],
				 'Turnover': np.log10(raw['close_turnover'])})
			cl = cl[cl.index.get_level_values('Symbol').isin(numstocks[n])]
			cl = cl.groupby(['Symbol', 'Percent']).mean()
			cl.reset_index(drop=False, inplace=True)

			fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
			sns.lineplot(x='Percent', y='Price Deviation', hue='Turnover', data=cl[cl['Percent'] <= limit],
					   linewidth=1, palette='YlOrRd', ax=ax1)
			ax1.xaxis.set_major_locator(ticker.MultipleLocator(1 / 100))
			ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
			ax1.set_xlabel("Removed liquidity")
			ax1.set_ylabel("Deviation from actual closing price [bps]")
			ax1.set_title("Price sensitivity of {} titles with respect to {} (N = {})"
					    .format(n, figdict[mode]['name'], str(len(numstocks[n]))))
			ax1.grid(which='both', axis='y')
			handles, labels = ax1.get_legend_handles_labels()
			ax1.legend(handles=handles[1:], labels=[round(float(l), 1) for l in labels[1:]],
					 loc=figdict[mode]['loc'], fontsize='small', title='log10(turnover)')
			fig.tight_layout()
			plt.savefig(figdir + "\\SensitivityFine\\Sens_Price_Percent_{}_{}".format(mode, n))
			# fig.show()
			plt.close()

			fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
			sns.lineplot(x='Percent', y='Volume Deviation', hue='Turnover', data=cl[cl['Percent'] <= limit],
					   linewidth=1, palette='YlGnBu', ax=ax1)
			ax1.xaxis.set_major_locator(ticker.MultipleLocator(1 / 100))
			ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
			ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
			ax1.set_ylim(top=0)
			ax1.set_xlabel("Removed liquidity")
			ax1.set_ylabel("Deviation from actual closing [\%]")
			ax1.set_title("Volume sensitivity of {} titles with respect to {} (N = {})"
					    .format(n, figdict[mode]['name'], str(len(numstocks[n]))))
			ax1.grid(which='major', axis='y')
			handles, labels = ax1.get_legend_handles_labels()
			ax1.legend(handles=handles[1:], labels=[round(float(l), 1) for l in labels[1:]],
					 loc=figdict[mode]['loc'], fontsize='small', title='log10(turnover)')
			fig.tight_layout()
			plt.savefig(figdir + "\\SensitivityFine\\Sens_Vol_Percent_{}_{}".format(mode, n))
			# fig.show()
			plt.close()

	def plt_cont_rmv_agg(self):
		"""
		Plot only used on the fine measurements of the data.
		"""
		# raw = copy.deepcopy(self._bcs_data.loc[['bid_limit', 'ask_limit', 'all_limit'], :])
		#
		# turn_df = pd.DataFrame({'turnover': raw['close_vol'] * raw['close_price'],
		# 				    'adj_turnover': raw['adj_vol'] * raw['adj_price']})
		#
		# df = pd.DataFrame({'Deviation': (raw['adj_price'] - raw['close_price']) / raw['close_price'] * 10 ** 4,
		# 			    'Volume': raw['adj_vol'],
		# 			    'Volume Delta': (raw['adj_vol'] - raw['close_vol']) / raw['close_vol'],
		# 			    'Turnover': turn_df['adj_turnover'] - turn_df['turnover'],
		# 			    'Turnover Delta': (turn_df['adj_turnover'] - turn_df['turnover']) / turn_df['turnover']
		# 			    }).reset_index()
		#
		# df.replace({'bid_limit': 'bid limit', 'ask_limit': 'ask limit', 'all_limit': 'all limit'},
		# 		 inplace=True)
		#
		# fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		# xmin, xmax = 0, 0.5
		# sns.lineplot(data=df, x='Percent', y='Deviation', hue='Mode',
		# 		   ax=ax1, ci=99)
		# ax1.grid(which='both')
		# ax1.hlines(0, xmin, xmax, 'k', lw=1)
		# ax1.set_xlim([xmin, xmax])
		# ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		# ax1.set_ylabel("Deviation in bps")
		# ax1.set_xlabel("Removed percentage of limit orders")
		# ax1.set_title(
		# 	"Average deviation of closing price depending on removed liquidity on SLI (Bootstrapped 99\% CI)")
		# fig.tight_layout()
		# plt.savefig(figdir + "\\SensitivityFine\\Avg_closeprice_SLI.png")
		# fig.show()
		# plt.close()
		#
		# fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		# sns.lineplot(data=df, x='Percent', y='Volume Delta', hue='Mode',
		# 		   ax=ax1, ci=99)
		# ax1.set_xlim([xmin, xmax])
		# ax1.hlines(0, xmin, xmax, 'k', lw=1)
		# ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		# ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		# ax1.grid(which='both')
		# ax1.set_ylabel("Deviation in \%")
		# ax1.set_xlabel("Removed percentage of limit orders")
		# ax1.set_title("Average deviation of volume depending on removed liquidity on SLI (Bootstrapped 99\% CI)")
		# fig.tight_layout()
		# plt.savefig(figdir + "\\SensitivityFine\\Avg_volume_dev_SLI.png")
		# fig.show()
		# plt.close()
		#
		# fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		# sns.lineplot(data=df, x='Percent', y='Turnover Delta', hue='Mode',
		# 		   ax=ax1, ci=99)
		# ax1.set_xlim([xmin, xmax])
		# ax1.hlines(0, xmin, xmax, 'k', lw=1)
		# ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		# ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		# ax1.grid(which='both')
		# ax1.set_ylabel("Deviation in \%")
		# ax1.set_xlabel("Removed percentage of limit orders")
		# ax1.set_title("Average deviation of turnover depending on removed liquidity on SLI (Bootstrapped 99\% CI)")
		# fig.tight_layout()
		# plt.savefig(figdir + "\\SensitivityFine\\Avg_turnover_dev_SLI.png")
		# fig.show()
		# plt.close()
		pass

	def plt_rmv_market_orders(self):
		"""Rough Sensitivity Data"""
		raw = self._bcs_data.loc['all_market', :]

		df = pd.DataFrame({'Deviation': (raw['adj_price'] - raw['close_price']) / raw['close_price'] * 10 ** 4,
					    'Turnover': raw['close_vol'] * raw['close_price'] / 10 ** 6,
					    'Volume Deviation': (raw['adj_vol'] - raw['close_vol']) / raw['close_vol']})
		df = df[abs(df['Deviation']) < 600]

		df = df.join(pd.Series(df['Turnover'].groupby('Symbol').mean(), name='Average Volume'), on='Symbol')

		df.reset_index(inplace=True)

		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(data=df, x='Date', y='Deviation', hue='Average Volume', sizes=(.5, 1.5),
				   size='Average Volume', palette='YlOrRd', ax=ax)
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles=handles[1:], labels=[round(float(l)) for l in labels[1:]],
				fontsize='small', title="Turnover (mn. CHF)", loc='upper left')
		ax.grid(which='major', axis='y')
		loca = dates.MonthLocator()
		form = dates.ConciseDateFormatter(loca)
		ax.xaxis.set_major_locator(loca)
		ax.xaxis.set_major_formatter(form)
		ax.set_ylabel("Deviation in bps")
		ax.set_xlabel("")
		ax.set_title("Deviation from original closing price when all market orders are removed by SLI title")
		fig.tight_layout()
		plt.savefig(figdir + "\\SensitivityRough\\Deviation_rmv_market")
		fig.show()
		plt.close()

		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(data=df, x='Date', y='Volume Deviation', hue='Average Volume', sizes=(.5, 1.5),
				   size='Average Volume', palette='YlOrRd', ax=ax)
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles=handles[1:], labels=[round(float(l)) for l in labels[1:]],
				fontsize='small', title="Turnover (mn. CHF)", loc='upper left')
		ax.grid(which='major', axis='y')
		ax.set_ylim([-1, 0])
		loca = dates.MonthLocator()
		form = dates.ConciseDateFormatter(loca)
		ax.xaxis.set_major_locator(loca)
		ax.xaxis.set_major_formatter(form)
		ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
		ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		ax.set_ylabel("Drop in traded closing volume")
		ax.set_xlabel("")
		ax.set_title("Decrease in number of shares traded when all market orders are removed by SLI title")
		fig.tight_layout()
		plt.savefig(figdir + "\\SensitivityRough\\Volume_rmv_market")
		fig.show()
		plt.close()


class DiscoAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._raw_data['close_turnover'] = self._raw_data['actual_close_price'] * self._raw_data['close_vol']
		self._raw_data.set_index(['Date', 'Symbol'], inplace=True)
		self._raw_vol_classify()

	def plt_closing_volume(self):
		limit = 50

		measdict = dict(shares={'call': 'Number of shares traded', 'data': self._avg_vol, 'col': 'close_vol'},
					 turnover={'call': 'Turnover', 'data': self._avg_turnover, 'col': 'close_turnover'})

		for m in iter(measdict.keys()):
			raw = self._raw_data
			avg_vol = measdict[m]['data']
			maxvol = avg_vol.max()
			tmp = pd.DataFrame({'Volume': raw[measdict[m]['col']] / maxvol}).groupby('Symbol').mean()
			tmp.sort_values('Volume', ascending=False, inplace=True)
			tmp.reset_index(inplace=True)

			fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
			sns.barplot(data=tmp[tmp['Volume'] > 0], x='Symbol', y='Volume', ax=ax, ci=None, palette='rainbow')
			ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
			ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
			ax.set_axisbelow(True)
			ax.grid(which='major', axis='both')
			ax.set_ylabel("{} divided by largest".format(measdict[m]['call']))
			ax.set_xlabel("Sorted titles")
			ax.set_yscale('log')
			ax.set_title('Average {} during closing auction sorted by title'.format(measdict[m]['call']))
			fig.tight_layout()
			plt.savefig(figdir + "\\PriceDiscovery\\{}_percent_aggregated".format(m))
			fig.show()
			plt.close(fig)

			fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
			sns.barplot(data=tmp.iloc[:limit, :], x='Symbol', y='Volume',
					  ax=ax, palette=sns.color_palette(def_color, 1))
			ax.set_axisbelow(True)
			ax.grid(which='major', axis='y')
			ax.set_ylabel("{} divided by largest".format(measdict[m]['call']))
			ax.set_title("Average {0} normalized for {1} most liquid titles".format(measdict[m]['call'], str(limit)))
			ax.tick_params(axis='x', rotation=90)
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			fig.tight_layout()
			plt.savefig(figdir + "\\PriceDiscovery\\{}_percent_stocks".format(m))
			fig.show()
			plt.close(fig)

	def plt_deviation_discovery(self):
		raw = copy.deepcopy(self._bcs_data)
		raw['pre_abs_spread'].replace({0: 0.01}, inplace=True)

		quants = raw.groupby('Date')['close_turnover']. \
			transform(lambda x: pd.qcut(x, 3, labels=['least liquid', 'neutral', 'most liquid']))
		quants.rename('Quantile', inplace=True)
		# quants.replace({1: 'least liquid', 2: 'neutral', 3: 'most liquid'}, inplace=True)
		df = pd.DataFrame(dict(dev=(raw['actual_close_price'] - raw['pre_midquote']) / raw['pre_midquote'] * 10 * 4,
						   dev_spread=(raw['actual_close_price'] - raw['pre_midquote']) / raw['pre_abs_spread'],
						   vol=raw['close_vol']))
		df = df.join(quants)
		df = df.reset_index()

		fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, dpi=dpi)
		xmin, xmax = df['Date'].iloc[0], df['Date'].iloc[-1]
		loca = dates.MonthLocator()
		form = dates.ConciseDateFormatter(loca)

		# Price Discovery Plot by Quantile
		for ax, qt in zip(axes, df['Quantile'].unique()):
			sns.lineplot(x='Date', y='dev', data=df[df['Quantile'] == qt], lw=1.2, ax=ax, ci=95)
			ax.hlines(0, xmin, xmax, 'k', lw=1)
			ax.set_ylim((-50, 50))
			ax.set_xlim((xmin, xmax))
			ax.set_ylabel("")
			ax.set_xlabel("")
			ax.set_title("Deviation of closing price to continuous midpoint for {} SLI titles (N = 10)".format(qt))
			ax.xaxis.set_major_locator(loca)
			ax.xaxis.set_major_formatter(form)

		axes[1].set_ylabel("Deviation [bps]")
		fig.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Closing_Deviations_terciles.png")
		# fig.show()
		plt.close(fig)

		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(x='Date', y='dev', data=df, ax=ax1, lw=1.2, ci=95)
		ax1.grid(which='major', axis='y')
		ax1.hlines(0, xmin, xmax, 'k', lw=1)
		ax1.xaxis.set_major_locator(loca)
		ax1.xaxis.set_major_formatter(form)
		ax1.set_ylim((-60, 60))
		ax1.set_xlim((xmin, xmax))
		ax1.set_xlabel("")
		ax1.set_ylabel("Deviation [bps]")
		ax1.set_title("Average daily deviation of closing price from last midpoint over SLI (Bootstrapped 95\% CI)")
		fig.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Closing_Deviations_aggregated")
		# fig.show()
		fig.clf()
		plt.close(fig)

		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(x='Date', y='dev', data=df, ax=ax1, lw=1.2, ci=None, estimator='std', hue='Quantile',
				   palette='cubehelix')
		ax1.grid(which='major', axis='y')
		ax1.xaxis.set_major_locator(loca)
		ax1.xaxis.set_major_formatter(form)
		ax1.set_yscale('log')
		# ax1.set_ylim(bottom=0.1)
		ax1.set_xlim((xmin, xmax))
		ax1.set_xlabel("")
		ax1.set_ylabel("Standard Deviation [bps]")
		ax1.set_title(
			"Standard deviation of closing price deviations from last midpoint over SLI (Bootstrapped 95\% CI)")
		fig.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Closing_Deviations_std_aggregated")
		fig.show()
		fig.clf()
		plt.close(fig)

		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(x='Date', y='dev_spread', data=df.dropna(), ax=ax1, lw=1.2, ci=None, estimator='std', hue='Quantile', palette='cubehelix')
		ax1.grid(which='major', axis='y')
		ax1.xaxis.set_major_locator(loca)
		ax1.xaxis.set_major_formatter(form)
		ax1.set_yscale('log')
		ax1.set_xlim((xmin, xmax))
		ax1.set_xlabel("")
		ax1.set_ylabel("Standard Deviation of dislocation divided by spread")
		ax1.set_title("Standard deviation of closing price deviations from last midpoint over SLI (Bootstrapped 95\% CI)")
		fig.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Closing_Deviations_std_spread_aggregated")
		fig.show()
		fig.clf()
		plt.close(fig)


		return df

	def plt_disco_distr_xsect(self, limit=15):
		# stocks = ['NESN','NOVN','ROG','UBSG','CSGN']
		stocks = self._avg_turnover.index[:20]
		raw = self._raw_data[self._raw_data.index.get_level_values(level='Symbol').isin(stocks)]

		df = pd.DataFrame({'Dev': (raw['actual_close_price'] - raw['pre_midquote']) / raw['pre_midquote'] * 10 ** 4,
					    'Vol': raw['close_turnover']})
		df = df.join(pd.Series(self._avg_turnover, name='Turnover'), on='Symbol', how='left')
		df.reset_index(inplace=True)
		ylim = 135
		df = df[abs(df['Dev']) <= ylim]
		print(df.dtypes)

		fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.boxplot(x='Symbol', y='Dev', palette='Set3', data=df,
				  ax=ax1, whis=[2.5, 97.5], linewidth=1, order=stocks)
		ax1.set_ylabel("Deviation from midquote in bps")
		ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))
		ax1.set_axisbelow(True)
		ax1.grid(which='major', axis='y', color='k', lw=0.7)
		ax1.set_ylim([-ylim, ylim])
		ax1.set_xlabel('')
		plt.xticks(rotation=90)
		ax1.set_title(
			"Distribution of deviations of closing from pre-close midquote of {} most \"closed\" stocks".format(
				limit))
		fig.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Distribution_xsection")
		fig.show()
		plt.close()

		return df

	def plt_disco_by_title(self, limit):
		titles = self._avg_turnover.index[:limit]
		raw = copy.deepcopy(self._bcs_data.loc[self._bcs_data.index.get_level_values(level='Symbol').isin(titles)])
		raw['pre_abs_spread'].replace({0: 0.01}, inplace=True)
		df = pd.DataFrame(
			{'Deviation': (raw['actual_close_price'] - raw['pre_midquote']) / raw['pre_midquote'] * 10 ** 4,
			 'Deviation Spread': (raw['actual_close_price'] - raw['pre_midquote']) / raw['pre_abs_spread']})
		df.reset_index(inplace=True)
		df = df[abs(df['Deviation']) < 300]

		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(data=df, x='Date', y='Deviation', hue='Symbol', palette='cubehelix', lw=1.2)
		ax.set_xlabel('')
		ax.set_ylabel('Deviation in bps')
		ax.set_title(
			'Deviation of closing price from last observed midquote for largest {} titles'.format(str(limit)))
		loca = dates.MonthLocator()
		ax.xaxis.set_major_locator(loca)
		ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(loca))
		ax.grid(which='major', axis='y')
		ax.set_axisbelow(True)
		fig.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Price_Deviation_Largest_Titles")
		fig.show()
		plt.close(fig)

		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
		sns.lineplot(data=df, x='Date', y='Deviation Spread', hue='Symbol', palette='cubehelix', lw=1.2)
		ax.set_xlabel('')
		ax.set_ylabel('Multiple of the pre-closing spread')
		ax.set_title(
			'Deviation of closing price from last observed midquote for largest {} titles'.format(str(limit)))
		loca = dates.MonthLocator()
		ax.set_axisbelow(True)
		ax.xaxis.set_major_locator(loca)
		ax.xaxis.set_major_formatter(dates.ConciseDateFormatter(loca))
		ax.grid(which='major', axis='y')
		fig.tight_layout()
		plt.savefig(figdir + "\\PriceDiscovery\\Spread_Deviation_Largest_Titles")
		fig.show()
		plt.close(fig)

		print(df.sort_values('Deviation Spread').head())


file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
#
# file_data = os.getcwd() + "\\Exports\\Sensitivity_rough_v1.csv"
# Sens = SensAnalysis(file_data, file_bcs)
# t = Sens.plt_rmv_limit_quant()

file_data = os.getcwd() + "\\Exports\\Price_Discovery_v1.csv"
Disco = DiscoAnalysis(file_data, file_bcs)
t = Disco.plt_deviation_discovery()
