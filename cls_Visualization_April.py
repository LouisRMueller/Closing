import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import os
import itertools

from matplotlib import pyplot as plt
from matplotlib import dates
from matplotlib import ticker
import seaborn as sns
import copy

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

register_matplotlib_converters()

pd.set_option('display.width', 300)
pd.set_option("display.max_columns", 12)
def_palette = "Set1"
sns.set_palette(def_palette, desat=0.8)
def_color = sns.color_palette(def_palette, 1)[0]


class Visualization:
	def __init__(self, datapath, bluechippath):
		self._bluechips = pd.read_csv(bluechippath)['symbol']
		self._raw_data = pd.read_csv(datapath, parse_dates=['Date'])

		self._figsize = (22 / 2.54, 13 / 2.54)
		self._cwd = os.getcwd() + "\\03 Presentation April"
		self._figdir = self._cwd + "\\Figures"
		self._lw = 1.2
		self._dpi = 300

		print("--- SuperClass Initiated ---")

	def _raw_vol_classify(self):
		self._bcs_data = self._raw_data[self._raw_data.index.get_level_values(level='Symbol').isin(self._bluechips)]
		self._avg_vol = self._raw_data['close_vol'].groupby('Symbol').mean().sort_values(ascending=False).dropna()
		self._avg_turnover = self._raw_data['close_turnover'].groupby('Symbol').mean().sort_values(
			ascending=False).dropna()

	def return_data(self):
		return self._raw_data


class SensVisual(Visualization):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._raw_data['close_turnover'] = self._raw_data['close_price'] * self._raw_data['close_vol']
		self._raw_data['Percent'] = self._raw_data['Percent'].round(2)
		self._raw_data.set_index(['Mode', 'Date', 'Symbol', 'Percent'], inplace=True)
		self._raw_vol_classify()

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
			colcount = df['Percent'].nunique()
			df['Percent'] = (df['Percent'] * 100).astype(int).astype(str) + " \%"

			fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			sns.lineplot(x='Date', y='Deviation', hue='Percent', data=df, ax=ax1, lw=1.25, ci=95, err_kws=dict(alpha=0.4),
					   palette=sns.color_palette(def_palette, colcount + 2, 0.8)[:colcount])
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
			plt.savefig(self._figdir + "\\SensitivityRough\\agg_remove_{}".format(mode))
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

			tmp = raw[['close_price', 'adj_price', 'close_vol', 'adj_vol']].join(quants, on=['Date', 'Symbol'], how='left')
			df = pd.DataFrame({'Price Deviation': (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10 ** 4,
						    'Volume Deviation': (tmp['adj_vol'] - tmp['close_vol']) / tmp['close_vol'],
						    'Quantile': tmp['quantile']}).reset_index()
			df = df[df['Percent'] <= limit]
			df['Percent'] = (df['Percent'] * 100).astype(int).astype(str) + '\%'

			fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			sns.boxplot(data=df, x='Percent', y='Price Deviation', palette='Reds', whis=[2.5, 97.5],
					  ax=ax1, hue='Quantile', showfliers=False, linewidth=1)
			ax1.set_xlabel('')
			ax1.set_title("Quantile distribution of closing price deviations when removing {}".format(namedict[mode]))
			ax1.set_ylabel("Deviation in bps")
			ax1.set_xlabel("Amount of liquidity removed")
			ax1.grid(which='major', axis='y')
			ax1.set_axisbelow(True)
			fig.tight_layout()
			plt.savefig(self._figdir + "\\SensitivityRough\\Quantile_distribution_{}".format(mode))
			fig.show()
			plt.close()

			fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			sns.boxplot(data=df, x='Percent', y='Volume Deviation', palette='Blues', whis=[2.5, 97.5],
					  ax=ax1, hue='Quantile', showfliers=False, linewidth=1)
			ax1.set_xlabel('')
			ax1.set_title("Quantile distribution of closing volume deviations when removing {}".format(namedict[mode]))
			ax1.set_ylabel("Deviation in \%")
			ax1.set_xlabel("Amount of liquidity removed")
			ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
			ax1.grid(which='major', axis='y')
			ax1.set_axisbelow(True)
			fig.tight_layout()
			plt.savefig(self._figdir + "\\SensitivityRough\\Quantile_distribution_volume_{}".format(mode))
			fig.show()
			plt.close()

	def plt_cont_rmv_indiv(self, mode):
		"""Only works with fine Sensitivity"""
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

			fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			sns.lineplot(x='Percent', y='Price Deviation', hue='Turnover', data=cl[cl['Percent'] <= limit], ax=ax1,
					   palette='Reds', lw=1.2)
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
			plt.savefig(self._figdir + "\\SensitivityFine\\Sens_Price_Percent_{}_{}".format(mode, n))
			fig.show()
			plt.close()

			fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			sns.lineplot(x='Percent', y='Volume Deviation', hue='Turnover', data=cl[cl['Percent'] <= limit],
					   palette='Blues', ax=ax1, lw=1.2)
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
			plt.savefig(self._figdir + "\\SensitivityFine\\Sens_Vol_Percent_{}_{}".format(mode, n))
			fig.show()
			plt.close()

	def plt_cont_rmv_indiv_v2(self, mode):
		"""Only works with fine Sensitivity"""
		limit = 0.3
		raw = copy.deepcopy(self._raw_data.loc[mode, :])
		raw = raw[raw['close_vol'] > 1000]

		stock_titles = self._bluechips[self._bluechips != 'DUFN']

		numstocks = {'SLI': stock_titles}
		figdict = dict(bid_limit=dict(name='bid', loc='lower left'),
					ask_limit=dict(name='ask', loc='upper left'),
					all_limit=dict(name='bid/ask', loc='upper left'))

		for n in numstocks.keys():
			cl = pd.DataFrame(
				{'Price Deviation': abs((raw['adj_price'] - raw['close_price']) / raw['close_price']) * 10 ** 4,
				 'Volume Deviation': (raw['adj_vol'] - raw['close_vol']) / raw['close_vol'],
				 'log10(turnover)': np.log10(raw['close_turnover'])})
			cl = cl[cl.index.get_level_values('Symbol').isin(numstocks[n])]
			cl = cl.groupby(['Symbol', 'Percent']).mean()
			cl.reset_index(drop=False, inplace=True)
			cl['Reducer'] = 'Average'

			fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			sns.lineplot(x='Percent', y='Price Deviation', hue='log10(turnover)', data=cl[cl['Percent'] <= limit], ax=ax1,
					   palette='Reds', lw=1.1, alpha=0.8)
			sns.lineplot(x='Percent', y='Price Deviation', data=cl[cl['Percent'] <= limit], ax=ax1,
					   lw=2.5, color='black', ci=None, markers='o', legend='full', label='Average', marker='o')
			ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
			ax1.xaxis.set_major_locator(ticker.MultipleLocator(5 / 100))
			ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
			ax1.set_xlabel("Percentage of removed {} liquidity".format(figdict[mode]['name']))
			ax1.set_ylabel("Absolute deviation from actual closing price [bps]")
			ax1.set_title("Price sensitivity of {} titles in 2019 with respect to removal of best {} limit orders (N = {})"
					    .format(n, figdict[mode]['name'], str(len(numstocks[n]))))
			ax1.grid(which='major', axis='both')
			ax1.set_xlim(left=0, right=limit)
			ax1.set_ylim(bottom=0, top=140)
			fig.tight_layout()
			plt.savefig(os.getcwd() + "\\02 Slides January\\Figures\\Sensitivity_{}_{}".format(mode, n))
			fig.show()
			plt.close()

	def plt_rmv_market_orders(self):
		"""Rough Sensitivity Data"""
		raw = self._bcs_data.loc['all_market', :]

		df = pd.DataFrame({'Deviation': (raw['adj_price'] - raw['close_price']) / raw['close_price'] * 10 ** 4,
					    'Turnover': raw['close_vol'] * raw['close_price'] / 10 ** 6,
					    'Volume Deviation': (raw['adj_vol'] - raw['close_vol']) / raw['close_vol']})
		df = df[abs(df['Deviation']) < 600]

		df = df.join(pd.Series(df['Turnover'].groupby('Symbol').mean(), name='Average Volume'), on='Symbol')

		df.reset_index(inplace=True)

		fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
		sns.lineplot(data=df, x='Date', y='Deviation', hue='Average Volume', sizes=(.5, 1.5),
				   size='Average Volume', palette='Reds', ax=ax)
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles=handles[1:], labels=[round(float(l)) for l in labels[1:]],
				fontsize='small', title="Turnover (mn. CHF)", loc='upper left')
		ax.grid(which='major', axis='y')
		loca = dates.MonthLocator()
		form = dates.ConciseDateFormatter(loca)
		ax.xaxis.set_major_locator(loca)
		ax.xaxis.set_major_formatter(form)
		ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
		ax.set_ylabel("Deviation in bps")
		ax.set_xlabel("")
		ax.set_title("Deviation from original closing price when all market orders are removed by SLI title")
		fig.tight_layout()
		plt.savefig(self._figdir + "\\SensitivityRough\\Deviation_rmv_market")
		fig.show()
		plt.close()

		fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
		sns.lineplot(data=df, x='Date', y='Volume Deviation', hue='Average Volume', sizes=(.5, 1.5),
				   size='Average Volume', palette='Blues', ax=ax)
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
		plt.savefig(self._figdir + "\\SensitivityRough\\Volume_rmv_market")
		fig.show()
		plt.close()


class DiscoVisual(Visualization):
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

			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			sns.barplot(data=tmp[tmp['Volume'] > 0], x='Symbol', y='Volume', ax=ax, ci=None,
					  palette=[sns.color_palette('Blues', 3, 0.7)[-1]])
			ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
			ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
			ax.set_axisbelow(True)
			ax.grid(which='major', axis='both')
			ax.set_ylabel("{} divided by largest".format(measdict[m]['call']))
			ax.set_xlabel("Sorted titles")
			ax.set_yscale('log')
			ax.set_title('Average {} during closing auction sorted by title'.format(measdict[m]['call']))
			fig.tight_layout()
			plt.savefig(self._figdir + "\\PriceDiscovery\\{}_percent_aggregated".format(m))
			fig.show()
			plt.close(fig)

			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			sns.barplot(data=tmp.iloc[:limit, :], x='Symbol', y='Volume', ax=ax,
					  palette=[sns.color_palette('Blues', 3, 0.7)[-1]])
			ax.set_axisbelow(True)
			ax.grid(which='major', axis='y')
			ax.set_ylabel("{} divided by largest".format(measdict[m]['call']))
			ax.set_xlabel('')
			ax.set_title("Average {0} normalized for {1} most liquid titles".format(measdict[m]['call'], str(limit)))
			ax.tick_params(axis='x', rotation=90)
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			fig.tight_layout()
			plt.savefig(self._figdir + "\\PriceDiscovery\\{}_percent_stocks".format(m))
			fig.show()
			plt.close(fig)

	def plt_deviation_discovery(self):
		raw = copy.deepcopy(self._bcs_data)
		raw['pre_abs_spread'].replace({0: 0.01}, inplace=True)

		quants = raw.groupby('Date')['close_turnover']. \
			transform(lambda x: pd.qcut(x, 3, labels=['least liquid', 'neutral', 'most liquid']))
		quants.rename('Quantile', inplace=True)
		df = pd.DataFrame(dict(dev=(raw['actual_close_price'] - raw['pre_midquote']) / raw['pre_midquote'] * 10 ** 4,
						   dev_spread=(raw['actual_close_price'] - raw['pre_midquote']) / raw['pre_abs_spread'],
						   vol=raw['close_vol']))
		df = df.join(quants)
		df = df.reset_index()

		fig, axes = plt.subplots(3, 1, figsize=self._figsize, sharex=True, dpi=self._dpi)
		xmin, xmax = df['Date'].iloc[0], df['Date'].iloc[-1]
		loca = dates.MonthLocator()
		form = dates.ConciseDateFormatter(loca)

		# Price Discovery Plot by Quantile
		for ax, qt in zip(axes, df['Quantile'].unique()):
			sns.lineplot(x='Date', y='dev', data=df[df['Quantile'] == qt], lw=1.2,
					   color=sns.color_palette('Reds', 4)[-1], ax=ax, ci=95)
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
		plt.savefig(self._figdir + "\\PriceDiscovery\\Closing_Deviations_terciles.png")
		fig.show()
		plt.close(fig)

		fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
		sns.lineplot(x='Date', y='dev', data=df, ax=ax1, lw=1.2, ci=95, color=sns.color_palette('Reds', 4)[-1])
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
		plt.savefig(self._figdir + "\\PriceDiscovery\\Closing_Deviations_aggregated")
		fig.show()
		fig.clf()
		plt.close(fig)

		fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
		sns.lineplot(x='Date', y='dev', data=df, ax=ax1, lw=1.2, ci=None, estimator='std', hue='Quantile',
				   palette=def_palette + '_r')
		ax1.grid(which='major', axis='y')
		ax1.xaxis.set_major_locator(loca)
		ax1.xaxis.set_major_formatter(form)
		ax1.set_yscale('log')
		ax1.set_xlim((xmin, xmax))
		ax1.set_xlabel("")
		ax1.set_ylabel("Standard Deviation [bps]")
		ax1.set_title(
			"Standard deviation of closing price deviations from last midpoint over SLI (Bootstrapped 95\% CI)")
		fig.tight_layout()
		plt.savefig(self._figdir + "\\PriceDiscovery\\Closing_Deviations_std_aggregated")
		fig.show()
		fig.clf()
		plt.close(fig)

		fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
		sns.lineplot(x='Date', y='dev_spread', data=df, ax=ax1, lw=1.2, ci=None, estimator='std',
				   hue='Quantile', palette=def_palette + '_r')
		ax1.grid(which='major', axis='y')
		ax1.xaxis.set_major_locator(loca)
		ax1.xaxis.set_major_formatter(form)
		ax1.set_yscale('log')
		ax1.set_xlim((xmin, xmax))
		ax1.set_xlabel("")
		ax1.set_ylabel("Standard Deviation of dislocation divided by spread")
		ax1.set_title("Standard deviation of closing price deviations from last midpoint over SLI (Bootstrapped 95\% CI)")
		fig.tight_layout()
		plt.savefig(self._figdir + "\\PriceDiscovery\\Closing_Deviations_std_spread_aggregated")
		# fig.show()
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

		fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
		sns.boxplot(x='Symbol', y='Dev', palette=[sns.color_palette('Reds', 5, 0.8)[-2]], data=df,
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
		plt.savefig(self._figdir + "\\PriceDiscovery\\Distribution_xsection")
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

		fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
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
		plt.savefig(self._figdir + "\\PriceDiscovery\\Price_Deviation_Largest_Titles")
		fig.show()
		plt.close(fig)

		fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
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
		plt.savefig(self._figdir + "\\PriceDiscovery\\Spread_Deviation_Largest_Titles")
		fig.show()
		plt.close(fig)


class IntervalVisual(Visualization):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)

		self._extract_indicators()
		super()._raw_vol_classify()

	def _extract_indicators(self):
		raw = self._raw_data
		raw['close_turnover'] = raw['close_price'] * raw['close_vol']
		raw['Absolute Imbalance'] = (raw['snap_bids'] - raw['snap_asks']) * raw['close_price']
		raw['Relative Imbalance'] = ((raw['snap_bids'] - raw['snap_asks']) / ((raw['snap_bids'] + raw['snap_asks']) / 2))
		raw['Turnover'] = raw['snap_price'] * raw['snap_vol']
		raw['Deviation'] = (raw['snap_price'] - raw['close_price']) / raw['close_price'] * 10000
		raw.set_index(['Date', 'Symbol', 'Lag'], inplace=True)

	def plot_months(self, save=False):
		def base_plot(linefunc, filename):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			linefunc(ax)
			ax.grid(which='major', axis='y')
			ax.set_xlabel("Seconds since start of auction")
			ax.legend(ncol=2, fontsize='small')
			ax.set_xlim([0, 600])
			ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\Intervals\\{1}".format(self._figdir, filename))
			plt.show()

		raw = self._raw_data.copy(deep=True)
		tmp = raw.groupby(['Lag', 'Date']).mean().reset_index()
		tmp['Date'] = tmp['Date'].apply(lambda d: d.strftime('%B'))
		tmp.rename(columns={'snap_OIB': 'Relative Imbalance', 'Date': 'Month'}, inplace=True)

		def volume_plot(ax):
			sns.lineplot(data=tmp, x='Lag', y='Turnover', hue='Month', ci=None, palette='cubehelix', lw=self._lw, ax=ax)
			ax.set_title("Average hypothetical closing turnover of SLI titles by month")
			ax.set_ylabel("Turnover [CHF]")

		def rel_OIB_plot(ax):
			sns.lineplot(data=tmp, x='Lag', y='Relative Imbalance', hue='Month', ci=None, palette='cubehelix', lw=self._lw, ax=ax)
			ax.axhline(0, c='k', lw=1.2)
			ax.set_ylim([-0.006, 0.006])
			ax.set_title("Average relative order imbalance of SLI titles by month")
			ax.set_ylabel("Relative order imbalance [\%]")
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

		def abs_OIB_plot(ax):
			ax.axhline(0, c='k', lw=1.2)
			sns.lineplot(data=tmp, x='Lag', y='Absolute Imbalance', hue='Month', ci=None, palette='cubehelix', lw=self._lw, ax=ax)
			ax.set_title("Average order imbalance of SLI titles by month")
			ax.set_ylabel("Order imbalance [CHF]")

		def price_plot(ax):
			ax.axhline(0, c='k', lw=1.2)
			sns.lineplot(data=tmp, x='Lag', y='Deviation', hue='Month', ci=None, palette='cubehelix', lw=self._lw, ax=ax)
			ax.set_title("Average price deviation SLI titles by month")
			ax.set_ylabel("Price deviation [bps]")

		base_plot(volume_plot, 'VolumePlotMonthly')
		base_plot(rel_OIB_plot, 'RelativeOIBMonthly')
		base_plot(abs_OIB_plot, 'AbsoluteOIBMonthly')
		base_plot(price_plot, 'PriceDeviationMonthly')

		return raw

	def plot_stocks_individual(self, nstocks=5, save=False):
		def base_plot(data, linefunc, stock, filename=None):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			linefunc(data, ax, stock)
			ax.grid(which='major', axis='y')
			ax.set_xlabel('')
			ax.legend(ncol=5, fontsize='small')
			ax.set_xlim(['2019-01-01', '2020-01-01'])
			locator = dates.MonthLocator(interval=1)
			formatter = dates.ConciseDateFormatter(locator, show_offset=False)
			ax.xaxis.set_major_locator(locator)
			ax.xaxis.set_major_formatter(formatter)
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\Intervals\\{1}_{2}".format(self._figdir, stock, filename))
			plt.close()

		stocks = self._avg_turnover.index[:nstocks].to_list()

		def deviation_plot(data, ax, stock):
			data = data[abs(data['Deviation']) < data['Deviation'].std() * 2]
			sns.lineplot(data=data, x='Date', y='Deviation', hue='Lag', lw=1,
					   ax=ax, ci=None, palette='cubehelix_r')
			ax.set_ylabel("Deviation from closing price [bps]")
			ax.set_title("Deviation of current price from closing price for {} (without outliers)".format(stock))

		def relative_OIB_plot(data, ax, stock):
			data = data[abs(data['Relative Imbalance']) < data['Relative Imbalance'].std() * 3]
			sns.lineplot(data=data, x='Date', y='Relative Imbalance', hue='Lag', lw=1,
					   ax=ax, ci=None, palette='cubehelix_r')
			ax.set_ylabel("Relative imbalance")
			ax.set_title("Relative order imbalance of current price from closing price for {} (without outliers".format(stock))
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

		def absolute_OIB_plot(data, ax, stock):
			# data['Absolute Imbalance'] = abs(data['Absolute Imbalance'])
			sns.lineplot(data=data, x='Date', y='Absolute Imbalance', hue='Lag', lw=1,
					   ax=ax, ci=None, palette='cubehelix_r')
			ax.set_ylabel("Absolute imbalance [CHF]")
			ax.set_title("Absolute order imbalance of current price from closing price for {}".format(stock))

		# ax.set_yscale('log')

		def turnover_plot(data, ax, stock):
			sns.lineplot(data=data, x='Date', y='Turnover', hue='Lag', lw=1,
					   ax=ax, ci=None, palette='cubehelix_r')
			ax.set_ylabel("Turnover in [CHF]")
			ax.set_title("Hypothetical turnover for {} by lag".format(stock))
			ax.set_ylim(bottom=0)

		def turnover_log_plot(data, ax, stock):
			sns.lineplot(data=data, x='Date', y='Turnover', hue='Lag', lw=1,
					   ax=ax, ci=None, palette='cubehelix_r')
			ax.set_ylabel("Turnover in [CHF]")
			ax.set_title("Hypothetical logarithmic turnover for {} by lag".format(stock))
			ax.set_yscale('log')

		for s in stocks:
			tmp = self._raw_data.xs(s, level='Symbol').reset_index(inplace=False)
			tmp['Lag'] = 'sec. ' + tmp['Lag'].astype(str)

			base_plot(tmp, deviation_plot, s, 'Deviation')
			base_plot(tmp, relative_OIB_plot, s, 'RelativeImbalance')
			base_plot(tmp, absolute_OIB_plot, s, 'AbsoluteImbalance')
			base_plot(tmp, turnover_plot, s, 'Turnover')
			base_plot(tmp, turnover_log_plot, s, 'Turnover_log')

		return tmp

	def plot_stocks_compare(self, nstocks=5, save=False):
		def base_plot(plotfunc, handle):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			plotfunc(ax)
			ax.grid(which='major', axis='y')
			ax.set_xlabel("Seconds since start of auction")
			ax.legend(ncol=2, fontsize='small')
			ax.set_xlim([0, 600])
			ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\Intervals\\Comparison\\{1}".format(self._figdir, handle))
			plt.show()

		stocks = self._avg_turnover.index[:nstocks].to_list()
		raw = self._raw_data.reset_index(inplace=False)
		raw = raw[raw['Symbol'].isin(stocks)]

		def turnover_plot(ax):
			sns.lineplot(data=raw, x='Lag', y='Turnover', hue='Symbol', lw=self._lw,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_ylim(bottom=0)

		def relative_OIB_plot(ax):
			ax.axhline(y=0, lw=1, c='k', ls='dashed')
			sns.lineplot(data=raw, x='Lag', y='Relative Imbalance', hue='Symbol', lw=self._lw,
					   ax=ax, palette='rainbow', hue_order=stocks, ci=None)
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

		def deviation_plot(ax):
			ax.axhline(y=0, lw=1, c='k', ls='dashed')
			sns.lineplot(data=raw, x='Lag', y='Deviation', hue='Symbol', lw=self._lw,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_ylabel('Average deviation from closing price [bps]')

		def deviation_conf_plot(ax):
			ax.axhline(y=0, lw=1, c='k', ls='dashed')
			sns.lineplot(data=raw, x='Lag', y='Deviation', hue='Symbol', lw=self._lw,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=95)
			ax.set_ylabel('Average deviation from closing price [bps]')

		base_plot(turnover_plot, 'Turnover')
		base_plot(relative_OIB_plot, 'RelativeTurnover')
		base_plot(deviation_plot, 'Deviation')
		base_plot(deviation_conf_plot, 'DeviationConf')

	def plot_stocks_within(self, nstocks=5, save=False):
		"""
		Very chaotic representation, but it indicates that many of the one-sided order imbalances are persistent.
		This probably comes from large rebalancing.
		"""

		def base_plot(plotfunc, stock, handle):
			tmp = raw[raw['Symbol'] == stock]
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			plotfunc(tmp, ax, stock)
			ax.grid(which='major', axis='y')
			ax.set_xlabel("Seconds since start of auction")
			# ax.legend(ncol=2, fontsize='small')
			ax.set_xlim([0, 600])
			# ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\Intervals\\Within\\{1}_{2}".format(self._figdir, handle, stock))
			plt.show()


		stocks = self._avg_turnover.index[:nstocks].to_list()
		raw = self._raw_data.reset_index(inplace=False)
		raw = raw[raw['Symbol'].isin(stocks)]

		def deviation_plot(data, ax, stock):
			sns.lineplot(data=data, x='Lag', y='Deviation', ax=ax, hue='Date', legend=None,
					   estimator=None, lw=0.5, palette='rainbow')
			ax.set_ylim([-500, 500])
			ax.set_ylabel("Deviation from closing price [bps]k")
			ax.set_title("{0}: Deviation of current price to closing price through auction".format(stock))

		def relative_OIB_plot(data, ax, stock):
			sns.lineplot(data=data, x='Lag', y='Relative Imbalance', ax=ax, units='Date', legend=None,
					   estimator=None, lw=0.5, palette='rainbow')
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_title("{0}: Relative order imbalance throughout auction".format(stock))
			ax.set_ylabel("Relative Imbalance [\%]")

		def absolute_OIB_plot(data, ax, stock):
			sns.lineplot(data=data, x='Lag', y='Absolute Imbalance', ax=ax, hue='Date',
					   estimator=None, lw=0.5, legend=None, palette='rainbow')
			ax.set_title("{0}: Absolute order imbalance throughout auction".format(stock))
			ax.set_ylabel("Absolute Imbalance [CHF]")

		for s in stocks:
			base_plot(deviation_plot, s, 'Deviation')
			base_plot(relative_OIB_plot, s, 'RelativeImbalance')
			base_plot(absolute_OIB_plot, s, 'AbsoluteImbalance')


	def plot_stocks_box(self, nstocks=5, save=False):
		def base_plot_box(plotfunc, stock, handle):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi, sharex=True, sharey=True)
			tmp = raw.loc[raw['Symbol'] == stock, :]
			plotfunc(tmp, ax, stock)
			ax.grid(which='major', axis='y')
			ax.set_xlabel("Seconds since start of auction")
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\Intervals\\Boxplots\\{1}_{2}".format(self._figdir, handle, stock))
			plt.show()

		stocks = self._avg_turnover.index[:nstocks]
		raw = self._raw_data.reset_index(inplace=False)
		raw = raw.loc[raw['Symbol'].isin(stocks.tolist()), :]
		raw.loc[:, 'Lag'] = raw.loc[:, 'Lag'].astype(object)


		def box_deviation_plot(data, ax, stock):
			sns.boxplot(data=data, x='Lag', y='Deviation', ax=ax, color=def_color)
			ax.set_ylim([-700, 700])
			ax.set_ylabel("Deviation [bps]")
			ax.set_title("{0}: Deviation from closing price by lag".format(stock))

		def box_relative_OIB_plot(data, ax, stock):
			sns.boxplot(data=data, x='Lag', y='Relative Imbalance', ax=ax, color=def_color)
			ax.set_ylim([-0.20, 0.20])
			ax.set_ylabel("Relative imbalance [\%]")
			ax.set_title("{0}: Deviation from closing price by lag".format(stock))
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

		for s in stocks:
			base_plot_box(box_deviation_plot, s, 'DeviationBox')
			base_plot_box(box_relative_OIB_plot, s, 'RelativeImbalanceBox')



########################################################################
file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
# mode, granularity = 'Sensitivity', 'rough'
# mode, granularity = 'Sensitivity', 'fine'
# mode, granularity = 'Discovery', None
mode, granularity = 'Intervals', None
file_data = os.getcwd() + "\\Exports\\Intervals_v3.csv"
Inter = IntervalVisual(file_data, file_bcs)
df = Inter.plot_stocks_within(save=True)
# df = Inter.plot_stocks_within(save=False)
