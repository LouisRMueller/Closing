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
pd.set_option("display.max_columns", 16)
def_palette = "Set1"
sns.set_palette(def_palette, desat=0.8)
def_color = sns.color_palette(def_palette, 1)[0]


class Visualization:
	def __init__(self, datapath):
		self._raw_data = pd.read_csv(datapath, parse_dates=['Date'])
		self._figsize = (22 / 2.54, 13 / 2.54)
		self._cwd = os.getcwd() + "\\03 Presentation April"
		self._figdir = self._cwd + "\\Figures"
		self._lw = 1.2
		self._dpi = 300

		print("--- SuperClass Initiated ---")

	def return_data(self):
		return self._raw_data


class SensVisual(Visualization):
	def __init__(self, datapath, base):
		super().__init__(datapath)
		self._base = base  # LimitVolume vs TotalVolume
		self._modes = dict(bid_limit="bid limit orders", ask_limit="ask limit orders", all_limit="all limit orders",
					    all_market="all market orders", cont_market="market orders from continuous phase")

		self._extract_factors()
		self._raw_data.set_index(['Mode', 'Date', 'Symbol', 'Percent'], inplace=True)
		print(self._raw_data.loc['all_market',:])

	def _extract_factors(self):
		data = self._raw_data
		data['Percent'] = data['Percent'].round(2)
		data['Closing turnover'] = data['close_price'] * data['close_vol']
		data['Deviation price'] = (data['adj_price'] - data['close_price']) / data['close_price'] * 10000
		data['Deviation turnover'] = (data['adj_vol'] - data['close_vol']) * data['close_price'] * 100

		self._avg_turnover = data.groupby('Symbol')['Closing turnover'].mean().sort_values(ascending=False).dropna()

	def plot_removed_time(self, save=False):
		"""Only works with rough percentages"""
		limit = 0.25

		def base_plot(funcdata, linefunc, filename):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.grid(which='major', axis='both')
			ax.axhline(0, c='k', lw=1)
			linefunc(funcdata, ax)
			# ax.legend(ncol=6, fontsize='small')
			loca = dates.MonthLocator()
			form = dates.ConciseDateFormatter(loca, show_offset=False)
			ax.xaxis.set_major_locator(loca)
			ax.xaxis.set_major_formatter(form)
			ax.set_axisbelow(True)
			fig.tight_layout()
			# if save:
			# 	plt.savefig(self._figdir + "\\SensitivityRough\\agg_remove_{}".format(mode))
			plt.show()

		def agg_price_fun(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='Deviation price', hue='Percent', ax=ax, lw=1.25, ci=None, err_kws=dict(alpha=0.4),
					   palette='magma')

		# EXECUTION
		for mode in iter(self._modes.keys()):
			df = self._raw_data.loc[mode, ['Deviation price', 'Deviation turnover']].reset_index(inplace=False)
			df = df[(df['Percent'] <= limit) & (df['Percent'] > 0)]
			colcount = df['Percent'].nunique()
			df['Percent'] = (df['Percent'] * 100).astype(int).astype(str) + " \%"

			base_plot(df, agg_price_fun, None)
			# fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			# ax1.set_axisbelow(True)
			# ax1.grid(which='major', axis='y')
			# ax1.set_xlabel("")
			# ax1.set_ylabel("Deviation from closing price in bps")
			# ax1.set_title(
			# 	"Average impact of gradual removal of {} across SLI (Bootstrapped 95\% CI)".format(namedict[mode]))

# def plt_rmv_limit_quant(self):
# 	limit = 0.2
# 	namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders",
# 				 all_limit="bid/ask limit orders")
#
# 	for mode in iter(namedict.keys()):
# 		raw = copy.deepcopy(self._bcs_data.loc[mode, :])
# 		quants = raw.groupby(['Date', 'Symbol']).first()
# 		quants = quants.groupby('Date')['close_turnover'].transform(lambda x: pd.qcut(x, 3, labels=range(1, 4)))
# 		quants.rename('quantile', inplace=True)
# 		quants.replace({1: 'least liquid', 2: 'neutral', 3: 'most liquid'}, inplace=True)
#
# 		tmp = raw[['close_price', 'adj_price', 'close_vol', 'adj_vol']].join(quants, on=['Date', 'Symbol'], how='left')
# 		df = pd.DataFrame({'Price Deviation': (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10 ** 4,
# 					    'Volume Deviation': (tmp['adj_vol'] - tmp['close_vol']) / tmp['close_vol'],
# 					    'Quantile': tmp['quantile']}).reset_index()
# 		df = df[df['Percent'] <= limit]
# 		df['Percent'] = (df['Percent'] * 100).astype(int).astype(str) + '\%'
#
# 		fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
# 		sns.boxplot(data=df, x='Percent', y='Price Deviation', palette='Reds', whis=[2.5, 97.5],
# 				  ax=ax1, hue='Quantile', showfliers=False, linewidth=1)
# 		ax1.set_xlabel('')
# 		ax1.set_title("Quantile distribution of closing price deviations when removing {}".format(namedict[mode]))
# 		ax1.set_ylabel("Deviation in bps")
# 		ax1.set_xlabel("Amount of liquidity removed")
# 		ax1.grid(which='major', axis='y')
# 		ax1.set_axisbelow(True)
# 		fig.tight_layout()
# 		plt.savefig(self._figdir + "\\SensitivityRough\\Quantile_distribution_{}".format(mode))
# 		fig.show()
# 		plt.close()
#
# 		fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
# 		sns.boxplot(data=df, x='Percent', y='Volume Deviation', palette='Blues', whis=[2.5, 97.5],
# 				  ax=ax1, hue='Quantile', showfliers=False, linewidth=1)
# 		ax1.set_xlabel('')
# 		ax1.set_title("Quantile distribution of closing volume deviations when removing {}".format(namedict[mode]))
# 		ax1.set_ylabel("Deviation in \%")
# 		ax1.set_xlabel("Amount of liquidity removed")
# 		ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# 		ax1.grid(which='major', axis='y')
# 		ax1.set_axisbelow(True)
# 		fig.tight_layout()
# 		plt.savefig(self._figdir + "\\SensitivityRough\\Quantile_distribution_volume_{}".format(mode))
# 		fig.show()
# 		plt.close()
#
# def plt_cont_rmv_indiv(self, mode):
# 	"""Only works with fine Sensitivity"""
# 	limit = 0.2
# 	raw = copy.deepcopy(self._raw_data.loc[mode, :])
# 	raw = raw[raw['close_vol'] > 1000]
# 	numstocks = {'available': self._avg_turnover.index[self._avg_turnover > 0],
# 			   'top 120': self._avg_turnover.index[:120],
# 			   'top 60': self._avg_turnover.index[:60],
# 			   'SLI': self._bluechips,
# 			   'top 20': self._avg_turnover.index[:20]}
# 	figdict = dict(bid_limit=dict(name='bid limit orders', loc='lower left'),
# 				ask_limit=dict(name='ask limit orders', loc='upper left'),
# 				all_limit=dict(name='bid/ask limit orders', loc='upper left'))
#
# 	for n in numstocks.keys():
# 		cl = pd.DataFrame(
# 			{'Price Deviation': (raw['adj_price'] - raw['close_price']) / raw['close_price'] * 10 ** 4,
# 			 'Volume Deviation': (raw['adj_vol'] - raw['close_vol']) / raw['close_vol'],
# 			 'Turnover': np.log10(raw['close_turnover'])})
# 		cl = cl[cl.index.get_level_values('Symbol').isin(numstocks[n])]
# 		cl = cl.groupby(['Symbol', 'Percent']).mean()
# 		cl.reset_index(drop=False, inplace=True)
#
# 		fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
# 		sns.lineplot(x='Percent', y='Price Deviation', hue='Turnover', data=cl[cl['Percent'] <= limit], ax=ax1,
# 				   palette='Reds', lw=1.2)
# 		ax1.xaxis.set_major_locator(ticker.MultipleLocator(1 / 100))
# 		ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# 		ax1.set_xlabel("Removed liquidity")
# 		ax1.set_ylabel("Deviation from actual closing price [bps]")
# 		ax1.set_title("Price sensitivity of {} titles with respect to {} (N = {})"
# 				    .format(n, figdict[mode]['name'], str(len(numstocks[n]))))
# 		ax1.grid(which='both', axis='y')
# 		handles, labels = ax1.get_legend_handles_labels()
# 		ax1.legend(handles=handles[1:], labels=[round(float(l), 1) for l in labels[1:]],
# 				 loc=figdict[mode]['loc'], fontsize='small', title='log10(turnover)')
# 		fig.tight_layout()
# 		plt.savefig(self._figdir + "\\SensitivityFine\\Sens_Price_Percent_{}_{}".format(mode, n))
# 		fig.show()
# 		plt.close()
#
# 		fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
# 		sns.lineplot(x='Percent', y='Volume Deviation', hue='Turnover', data=cl[cl['Percent'] <= limit],
# 				   palette='Blues', ax=ax1, lw=1.2)
# 		ax1.xaxis.set_major_locator(ticker.MultipleLocator(1 / 100))
# 		ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# 		ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# 		ax1.set_ylim(top=0)
# 		ax1.set_xlabel("Removed liquidity")
# 		ax1.set_ylabel("Deviation from actual closing [\%]")
# 		ax1.set_title("Volume sensitivity of {} titles with respect to {} (N = {})"
# 				    .format(n, figdict[mode]['name'], str(len(numstocks[n]))))
# 		ax1.grid(which='major', axis='y')
# 		handles, labels = ax1.get_legend_handles_labels()
# 		ax1.legend(handles=handles[1:], labels=[round(float(l), 1) for l in labels[1:]],
# 				 loc=figdict[mode]['loc'], fontsize='small', title='log10(turnover)')
# 		fig.tight_layout()
# 		plt.savefig(self._figdir + "\\SensitivityFine\\Sens_Vol_Percent_{}_{}".format(mode, n))
# 		fig.show()
# 		plt.close()
#
# def plt_cont_rmv_indiv_v2(self, mode):
# 	"""Only works with fine Sensitivity"""
# 	limit = 0.3
# 	raw = copy.deepcopy(self._raw_data.loc[mode, :])
# 	raw = raw[raw['close_vol'] > 1000]
#
# 	stock_titles = self._bluechips[self._bluechips != 'DUFN']
#
# 	numstocks = {'SLI': stock_titles}
# 	figdict = dict(bid_limit=dict(name='bid', loc='lower left'),
# 				ask_limit=dict(name='ask', loc='upper left'),
# 				all_limit=dict(name='bid/ask', loc='upper left'))
#
# 	for n in numstocks.keys():
# 		cl = pd.DataFrame(
# 			{'Price Deviation': abs((raw['adj_price'] - raw['close_price']) / raw['close_price']) * 10 ** 4,
# 			 'Volume Deviation': (raw['adj_vol'] - raw['close_vol']) / raw['close_vol'],
# 			 'log10(turnover)': np.log10(raw['close_turnover'])})
# 		cl = cl[cl.index.get_level_values('Symbol').isin(numstocks[n])]
# 		cl = cl.groupby(['Symbol', 'Percent']).mean()
# 		cl.reset_index(drop=False, inplace=True)
# 		cl['Reducer'] = 'Average'
#
# 		fig, ax1 = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
# 		sns.lineplot(x='Percent', y='Price Deviation', hue='log10(turnover)', data=cl[cl['Percent'] <= limit], ax=ax1,
# 				   palette='Reds', lw=1.1, alpha=0.8)
# 		sns.lineplot(x='Percent', y='Price Deviation', data=cl[cl['Percent'] <= limit], ax=ax1,
# 				   lw=2.5, color='black', ci=None, markers='o', legend='full', label='Average', marker='o')
# 		ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
# 		ax1.xaxis.set_major_locator(ticker.MultipleLocator(5 / 100))
# 		ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
# 		ax1.set_xlabel("Percentage of removed {} liquidity".format(figdict[mode]['name']))
# 		ax1.set_ylabel("Absolute deviation from actual closing price [bps]")
# 		ax1.set_title("Price sensitivity of {} titles in 2019 with respect to removal of best {} limit orders (N = {})"
# 				    .format(n, figdict[mode]['name'], str(len(numstocks[n]))))
# 		ax1.grid(which='major', axis='both')
# 		ax1.set_xlim(left=0, right=limit)
# 		ax1.set_ylim(bottom=0, top=140)
# 		fig.tight_layout()
# 		plt.savefig(os.getcwd() + "\\02 Slides January\\Figures\\Sensitivity_{}_{}".format(mode, n))
# 		fig.show()
# 		plt.close()
#
# def plt_rmv_market_orders(self):
# 	"""Rough Sensitivity Data"""
# 	raw = self._bcs_data.loc['all_market', :]
#
# 	df = pd.DataFrame({'Deviation': (raw['adj_price'] - raw['close_price']) / raw['close_price'] * 10 ** 4,
# 				    'Turnover': raw['close_vol'] * raw['close_price'] / 10 ** 6,
# 				    'Volume Deviation': (raw['adj_vol'] - raw['close_vol']) / raw['close_vol']})
# 	df = df[abs(df['Deviation']) < 600]
#
# 	df = df.join(pd.Series(df['Turnover'].groupby('Symbol').mean(), name='Average Volume'), on='Symbol')
#
# 	df.reset_index(inplace=True)
#
# 	fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
# 	sns.lineplot(data=df, x='Date', y='Deviation', hue='Average Volume', sizes=(.5, 1.5),
# 			   size='Average Volume', palette='Reds', ax=ax)
# 	handles, labels = ax.get_legend_handles_labels()
# 	ax.legend(handles=handles[1:], labels=[round(float(l)) for l in labels[1:]],
# 			fontsize='small', title="Turnover (mn. CHF)", loc='upper left')
# 	ax.grid(which='major', axis='y')
# 	loca = dates.MonthLocator()
# 	form = dates.ConciseDateFormatter(loca)
# 	ax.xaxis.set_major_locator(loca)
# 	ax.xaxis.set_major_formatter(form)
# 	ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
# 	ax.set_ylabel("Deviation in bps")
# 	ax.set_xlabel("")
# 	ax.set_title("Deviation from original closing price when all market orders are removed by SLI title")
# 	fig.tight_layout()
# 	plt.savefig(self._figdir + "\\SensitivityRough\\Deviation_rmv_market")
# 	fig.show()
# 	plt.close()
#
# 	fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
# 	sns.lineplot(data=df, x='Date', y='Volume Deviation', hue='Average Volume', sizes=(.5, 1.5),
# 			   size='Average Volume', palette='Blues', ax=ax)
# 	handles, labels = ax.get_legend_handles_labels()
# 	ax.legend(handles=handles[1:], labels=[round(float(l)) for l in labels[1:]],
# 			fontsize='small', title="Turnover (mn. CHF)", loc='upper left')
# 	ax.grid(which='major', axis='y')
# 	ax.set_ylim([-1, 0])
# 	loca = dates.MonthLocator()
# 	form = dates.ConciseDateFormatter(loca)
# 	ax.xaxis.set_major_locator(loca)
# 	ax.xaxis.set_major_formatter(form)
# 	ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
# 	ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
# 	ax.set_ylabel("Drop in traded closing volume")
# 	ax.set_xlabel("")
# 	ax.set_title("Decrease in number of shares traded when all market orders are removed by SLI title")
# 	fig.tight_layout()
# 	plt.savefig(self._figdir + "\\SensitivityRough\\Volume_rmv_market")
# 	fig.show()
# 	plt.close()


class DiscoVisual(Visualization):
	def __init__(self, datapath, returnpath):
		super().__init__(datapath)
		self._raw_data.set_index(['Date', 'Symbol'], inplace=True)
		self._returns = pd.read_csv(returnpath, index_col=['Date', 'Symbol'], parse_dates=['Date'])
		self._calculate_factors()
		self._close_volume_classify()

	def _calculate_factors(self):
		data = self._raw_data
		rets = self._returns

		data['close_turnover'] = data['actual_close_price'] * data['close_vol']
		data['Closing Return'] = (data['actual_close_price'] - data['pre_midquote']) / data['pre_midquote']
		data['Absolute Imbalance'] = (data['start_bids'] - data['start_asks']) * data['close_price'] / 10 ** 6
		data['Relative Imbalance'] = (data['start_bids'] - data['start_asks']) / (data['start_bids'] + data['start_asks'])

		# Add liquidity quantiles
		quants = data.groupby('Date')['close_turnover'].transform(lambda x: pd.qcut(x, 3, labels=range(1, 4)))
		quants.rename('Volume quantile', inplace=True)
		quants.replace({1: 'least volume', 2: 'neutral', 3: 'most volume'}, inplace=True)
		self._raw_data = data.join(quants, on=['Date', 'Symbol'])

		# Calculate WPDC
		rets = rets.join(rets['return_open_close'].abs().groupby('Date').sum(), rsuffix='_abs_total', on='Date')
		rets['PDC'] = rets['return_close'] / rets['return_open_close']
		rets['WPDC'] = rets['PDC'] * abs(rets['return_open_close'] / rets['return_open_close_abs_total'])
		rets = rets.join(rets['WPDC'].groupby('Date').sum(), rsuffix='day', on='Date')
		self._returns = rets.join(quants)

	def plot_oib_returns(self, save):
		def base_plot(funcdata, linefunc, filename):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.grid(which='major', axis='both')
			ax.axvline(0, c='k', lw=1)
			linefunc(funcdata, ax)
			ax.legend(ncol=6, fontsize='small')
			ax.set_ylabel('Return [\%]')
			ax.set_ylim([-0.025, 0.025])
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_axisbelow(True)
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\PriceDiscovery\\{1}".format(self._figdir, filename))
			plt.show()

		data = self._raw_data
		returns = self._returns

		def abs_oib_returns(tmp, ax):
			sns.scatterplot(data=tmp, x='Absolute Imbalance', y='Closing Return', ax=ax, hue='Volume quantile',
						 ec='k', palette='cubehelix')
			ax.set_title("Full order imbalance versus closing return by closing volume tercile")
			ax.set_xlabel("Absolute imbalance at auction start [CHF]")
			xlimit = min(abs(min(tmp['Absolute Imbalance'])), max(tmp['Absolute Imbalance'])) * 1.4
			ax.set_xlim([-xlimit, xlimit])

		def rel_oib_returns(tmp, ax):  # Only marginally useful
			sns.scatterplot(data=tmp, x='Relative Imbalance', y='Closing Return', ax=ax, hue='Volume quantile',
						 ec='k', palette='cubehelix')
			ax.set_title("Relative order imbalance versus closing return by closing volume tercile")
			ax.set_xlabel("Relative order imbalance at auction start [\%]")
			ax.set_xlim([-1, 1])
			ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

		base_plot(data, abs_oib_returns, 'AbsoluteOIB_Returns')
		base_plot(data, rel_oib_returns, 'RelativeOIB_Returns')

		return returns

	def wpdc_plots(self, save):
		def base_plot(funcdata, axfunc, filename):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.grid(which='major', axis='both')
			ax.axhline(0, c='k', lw=1)
			axfunc(funcdata, ax)
			ax.legend(ncol=4, fontsize='x-small')
			ax.set_ylabel('WPDC')
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_axisbelow(True)
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\PriceDiscovery\\{1}".format(self._figdir, filename))
			plt.show()

		data = self._raw_data
		returns = self._returns
		wpdc_oib_df = data[['Relative Imbalance', 'close_turnover']].join(returns).reset_index()

		def wpdc_time_plot(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='WPDC', ax=ax, hue='Volume quantile', estimator='sum',
					   ci=None, lw=1.1, palette='cubehelix', marker='.', mew=0, ms=8)
			ax.set_title("Weighted price discovery contribution per day over time in 2019")
			ax.set_xlabel("")
			ax.set_xlim([pd.datetime(2019, 1, 1), pd.datetime(2020, 1, 1)])
			locator = dates.MonthLocator()
			formatter = dates.ConciseDateFormatter(locator, show_offset=False)
			ax.xaxis.set_major_locator(locator)
			ax.xaxis.set_major_formatter(formatter)

		def wpdc_oib_stockday_plot(funcdata, ax):
			sns.scatterplot(data=funcdata, x='Relative Imbalance', y='WPDC', ax=ax, hue='Volume quantile',
						 ec='k', palette='cubehelix')
			ax.set_title("Weighted price discovery contribution and relative order imbalance by stockday")
			ax.set_xlabel("Relative Imbalance")
			ax.axvline(0, c='k', lw=1)

		def wpdc_oib_stock_plot(funcdata, ax):
			funtmp = funcdata[funcdata['return_open_close'] != 0]
			funtmp = funtmp.groupby('Symbol').median().reset_index()
			sns.scatterplot(data=funtmp, x='Relative Imbalance', y='PDC', ax=ax, size='Symbol', hue='Symbol', ec='k',
						 palette='cubehelix_r', hue_order=self._avg_turnover.index,
						 sizes=(10, 400), size_order=self._avg_turnover.index)
			ax.set_title("Weighted price discovery contribution and relative order imbalance by stock")
			ax.set_xlabel("Relative Imbalance")
			# ax.legend(loc='lower right')
			ax.set_xlim(right=0.6)
			ax.set_ylim(top=0.1, bottom=-0.1)
			ax.axvline(0, c='k', lw=1)

		def wpdc_oib_day_plot(funcdata, ax):
			funtmp = funcdata[funcdata['return_open_close'] != 0]
			funtmp.loc[:, 'Date'] = "grt" + (funtmp['Date'].dt.dayofyear).astype(str)
			vols = funtmp.groupby('Date').sum()['close_turnover']
			vols = (vols / vols.max() * 400).to_dict()
			funtmp = funtmp.groupby('Date').mean().reset_index()
			funtmp.rename(columns={'close_turnover': 'Turnover'}, inplace=True)

			sns.scatterplot(data=funtmp, x='Relative Imbalance', y='WPDCday', ax=ax, size='Turnover',
						 ec='k', legend=None, sizes=(5, 500))
			ax.axvline(0, c='k', lw=1)
			ax.set_title("Weighted price discovery contribution and relative order imbalance by day")
			ax.set_xlabel("Relative Imbalance")
			ax.set_xlim([-.22, .22])
			ax.set_ylim([-.4, .4])

		base_plot(returns.reset_index(), wpdc_time_plot, 'WPDC_time')
		base_plot(wpdc_oib_df, wpdc_oib_stockday_plot, 'WPDC_OIB_stockday')
		base_plot(wpdc_oib_df, wpdc_oib_stock_plot, 'WPDC_OIB_stock')
		base_plot(wpdc_oib_df, wpdc_oib_day_plot, 'WPDC_OIB_day')


class IntervalVisual(Visualization):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)

		self._extract_indicators()
		self._close_volume_classify()

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

		def rel_oib_plot(ax):
			sns.lineplot(data=tmp, x='Lag', y='Relative Imbalance', hue='Month', ci=None, palette='cubehelix', lw=self._lw, ax=ax)
			ax.axhline(0, c='k', lw=1.2)
			ax.set_ylim([-0.006, 0.006])
			ax.set_title("Average relative order imbalance of SLI titles by month")
			ax.set_ylabel("Relative order imbalance [\%]")
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

		def abs_oib_plot(ax):
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
		base_plot(rel_oib_plot, 'RelativeOIBMonthly')
		base_plot(abs_oib_plot, 'AbsoluteOIBMonthly')
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
# file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
# file_rets = os.getcwd() + "\\Data\\end_of_day_returns_bluechips.csv"
granularity = 'rough'
file_data = os.getcwd() + "\\Exports\\Sensitivity_{}_LimitOrders_v3.csv".format(granularity)
Sens = SensVisual(file_data, 'Volume')
# df = Sens.plot_removed_time()

