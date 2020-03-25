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
		self._dpi = 300

		print("--- SuperClass Initiated ---")

	def return_data(self):
		return self._raw_data


class SensVisual(Visualization):
	def __init__(self, datapath: str, base: str, limit: float = 0.3):
		super().__init__(datapath)
		self._manipulate_data()
		self._raw_data.set_index(['Mode', 'Date', 'Symbol', 'Percent'], inplace=True)
		self._raw_data.drop(index=['ALC'], level='Symbol', inplace=True)
		self._define_quantiles()

		# Attributes
		if base in {'SeparateOrders', 'FullLiquidity', 'CrossedVolume'}:
			self._base = base  # LimitOrders vs TotalVolume
		else:
			raise ValueError("base not in {'SeparateOrders','FullLiquidity','CrossedVolume'}")

		self._base_d = dict(SeparateOrders='Separate liquidity basis',
						FullLiquidity='Full liquidity basis',
						CrossedVolume='Execution volume basis')

		self._mode_d = dict(bid_limit="bid orders", ask_limit="ask orders", all_limit="bid/ask orders",
						all_market="all market orders", cont_market="market orders (cont. phase)")
		self._limit = limit

	def _manipulate_data(self):
		data = self._raw_data
		data['Percent'] = data['Percent'].round(2)
		data['Closing turnover'] = data['close_price'] * data['close_vol']
		data['Deviation price'] = (data['adj_price'] - data['close_price']) / data['close_price'] * 10000
		data['Deviation turnover'] = (data['adj_vol'] - data['close_vol']) / data['close_vol']

		# Absolute deviations for symmetric removals
		sel = data.loc[data['Mode'].isin(['all_limit', 'all_market', 'cont_market']), 'Deviation price']
		data.loc[data['Mode'].isin(['all_limit', 'all_market', 'cont_market']), 'Deviation price'] = sel.abs()

		self._avg_turnover = data.groupby('Symbol')['Closing turnover'].mean().sort_values(ascending=False).dropna()

	def _define_quantiles(self):
		data = self._raw_data.sort_index()
		quants = data['Closing turnover'].groupby(['Mode', 'Date', 'Symbol']).first()
		quants = quants.groupby('Date').transform(lambda x: pd.qcut(x, 3, labels=['least liquid', 'neutral', 'most liquid']))
		quants.rename('Volume quantile', inplace=True)

		self._raw_data = data.join(quants)

	def plot_removal_time(self, save: bool = False, show: bool = True) -> None:
		"""Only works with rough percentages"""
		b = self._base
		data = self._raw_data
		limit = 0.6

		def base_plot(funcdata, linefunc, filename):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.grid(which='major', axis='y')
			ax.axhline(0, c='k', lw=1.5)
			linefunc(funcdata, ax)
			ax.legend(ncol=2, fontsize='small')
			ax.set_xlim([pd.datetime(2019, 1, 1), pd.datetime(2020, 1, 1)])
			ax.set_xlabel("")

			loca = dates.MonthLocator()
			form = dates.ConciseDateFormatter(loca, show_offset=False)
			ax.xaxis.set_major_locator(loca)
			ax.xaxis.set_major_formatter(form)
			ax.set_axisbelow(True)
			fig.tight_layout()
			if save:
				plt.savefig(self._figdir + "\\Sensitivity\\{b}\\remove_{n}_{m}".format(n=filename, m=mode, b=b),
						  transparent=True)
			plt.show() if show else plt.close()

		def agg_price_fun(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='Deviation price', hue='Percent', estimator='mean',
					   ax=ax, lw=0.9, ci=None, err_kws=dict(alpha=0.2), palette='Reds_d')
			ax.set_ylabel("Deviation of closing price [bps]")
			ax.set_title("[{b}] Average effect of removal of {m} on closing price".format(b=self._base_d[b], m=self._mode_d[mode]))

		def agg_volume_fun(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='Deviation turnover', hue='Percent', estimator='mean',
					   ax=ax, lw=0.9, ci=None, err_kws=dict(alpha=0.2), palette='Blues_d')
			ax.set_ylabel("Deviation of closing volume [\%]")
			ax.set_title("[{b}] Average effect of removal of {m} on closing volume".format(b=self._base_d[b], m=self._mode_d[mode]))
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
			ax.set_ylim(top=0, bottom=-1)

		def agg_price_mkt_fun(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='Deviation price', hue='Volume quantile', estimator='mean',
					   ax=ax, lw=1.2, ci=None, err_kws=dict(alpha=0.2), palette='Reds')
			ax.set_ylabel("Deviation of closing price [bps]")
			ax.set_title("Average absolute effect of removal of {m} on closing price".format(m=self._mode_d[mode]))

		def agg_volume_mkt_fun(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='Deviation turnover', hue='Volume quantile', estimator='mean',
					   ax=ax, lw=1.2, ci=None, err_kws=dict(alpha=0.2), palette='Blues')
			ax.set_ylabel("Deviation of closing volume [\%]")
			ax.set_title("Average absolute effect of removal of {m} on closing volume".format(m=self._mode_d[mode]))
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_ylim(top=0, bottom=-1)

		# EXECUTION
		for mode in iter(self._mode_d.keys()):
			df = data.loc[mode, ['Deviation price', 'Deviation turnover', 'Volume quantile']].reset_index(inplace=False)
			df = df[((df['Percent'] <= limit) & (df['Percent'] > 0)) | (df['Percent'] == 1)]
			df['Percent'] = (df['Percent'] * 100).astype(int).astype(str) + "\%"

			if mode in {'all_market', 'cont_market'}:
				base_plot(df, agg_price_mkt_fun, 'Price')
				base_plot(df, agg_volume_mkt_fun, 'Turnover')
			else:
				base_plot(df, agg_price_fun, 'Price')
				base_plot(df, agg_volume_fun, 'Turnover')

	def plot_removal_quantiles(self, save: bool = False, show: bool = True) -> None:
		limit = 0.35
		b = self._base
		data = self._raw_data[['Deviation price', 'Deviation turnover', 'Volume quantile']]

		def base_plot(funcdata, linefunc, filename):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.grid(which='major', axis='y')
			ax.axhline(0, c='k', lw=1)
			linefunc(funcdata, ax)
			ax.legend(fontsize='small')
			ax.set_axisbelow(True)
			fig.tight_layout()
			if save:
				plt.savefig(self._figdir + "\\Sensitivity\\{b}\\quantile_removal_{n}_{m}".format(n=filename, m=mode, b=b),
						  transparent=True)
			plt.show() if show else plt.close()

		def quant_price_func(funcdata, ax):
			sns.boxplot(data=funcdata[funcdata['Percent'] <= limit], ax=ax, x='PercentStr', y='Deviation price',
					  hue='Volume quantile', showfliers=False, whis=[2.5, 97.5], palette='Reds', linewidth=1)
			ax.set_ylabel("Price deviation [bps]")
			ax.set_title("[{b}] Quantile distribution of changes in closing price by removing {m}".format(b=self._base_d[b], m=self._mode_d[mode]))
			ax.set_xlabel("Removed Liquidity [\%]")

		def quant_vol_func(funcdata, ax):
			sns.boxplot(data=funcdata, ax=ax, x='PercentStr', y='Deviation turnover', hue='Volume quantile',
					  showfliers=False, whis=[2.5, 97.5], palette='Blues', linewidth=1)
			ax.set_ylabel("Turnover deviation [\%]")
			ax.set_title("[{b}] Quantile distrubution of changes in closing turnover by removing {m}".format(b=self._base_d[b], m=self._mode_d[mode]))
			ax.set_xlabel("Removed Liquidity [\%]")
			ax.set_ylim([-1.05, 0.05])
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

		def quant_price_mkt_func(funcdata, ax):
			sns.boxplot(data=funcdata, ax=ax, x='Mode', y='Deviation price',
					  hue='Volume quantile', showfliers=False, whis=[2.5, 97.5], palette='Reds', linewidth=1)
			ax.set_ylabel("Price deviation [bps]")
			ax.set_xlabel("")
			ax.set_title("Quantile distribution of changes in closing price by removing market orders".format(b=self._base_d[b]))
			mode = 'market'

		def quant_vol_mkt_func(funcdata, ax):
			sns.boxplot(data=funcdata, ax=ax, x='Mode', y='Deviation turnover', hue='Volume quantile',
					  showfliers=False, whis=[2.5, 97.5], palette='Blues', linewidth=1)
			ax.set_ylabel("Turnover deviation [\%]")
			ax.set_title("Quantile distrubution of changes in closing turnover by removing market orders".format(b=self._base_d[b]))
			ax.set_xlabel("")
			ax.set_ylim([-1.05, 0.05])
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			mode = 'market'

		for mode in iter(self._mode_d.keys()):
			tmp = data.loc[mode, :].reset_index(inplace=False)
			tmp = tmp[tmp['Percent'] > 0]
			tmp['PercentStr'] = (tmp['Percent'] * 100).astype(int).astype(str) + '\%'
			if mode not in {'all_market', 'cont_market'}:
				base_plot(tmp, quant_price_func, 'Price')
				base_plot(tmp, quant_vol_func, 'Turnover')

		tmp2 = data.loc[['all_market', 'cont_market']].reset_index(inplace=False)
		tmp2['Mode'].replace(self._mode_d, inplace=True)
		mode = 'market'
		base_plot(tmp2, quant_price_mkt_func, 'Price')
		base_plot(tmp2, quant_vol_mkt_func, 'Turnover')


class DiscoVisual(Visualization):
	def __init__(self, datapath, returnpath):
		super().__init__(datapath)
		self._raw_data.set_index(['Date', 'Symbol'], inplace=True)
		self._returns = pd.read_csv(returnpath, index_col=['Date', 'Symbol'], parse_dates=['Date'])
		self._calculate_factors()
		self._define_quantiles()

		# Attributes
		self._lw = 1.2
		self._ms = 8
		self._avg_turnover = self._raw_data['Close turnover'].groupby('Symbol').mean().sort_values(ascending=False)

	def _calculate_factors(self):
		data = self._raw_data
		rets = self._returns

		data['Close turnover'] = data['actual_close_price'] * data['close_vol'] / 10 ** 6
		data['Closing Return'] = (data['actual_close_price'] - data['pre_midquote']) / data['pre_midquote']
		data['Absolute Imbalance'] = (data['start_bids'] - data['start_asks']) * data['close_price'] / 10 ** 6
		data['Relative Imbalance'] = (data['start_bids'] - data['start_asks']) / (data['start_bids'] + data['start_asks'])

		# Calculate WPDC
		rets = rets.join(rets['return_open_close'].abs().groupby('Date').sum(), rsuffix='_abs_total', on='Date')
		rets['PDC'] = rets['return_close'] / rets['return_open_close']
		rets['WPDC'] = rets['PDC'] * abs(rets['return_open_close'] / rets['return_open_close_abs_total'])
		self._returns = rets.join(rets['WPDC'].groupby('Date').sum(), rsuffix='day', on='Date')

	def _define_quantiles(self):
		data = self._raw_data.sort_index()
		rets = self._returns.sort_index()

		quants = data['Close turnover'].groupby('Date').transform(
			lambda x: pd.qcut(x, 3, labels=['least liquid', 'neutral', 'most liquid']))
		quants.rename('Volume quantile', inplace=True)
		self._raw_data = data.join(quants)
		self._returns = rets.join(quants)

	def plot_oib_returns(self, save: bool = False, show: bool = True) -> None:
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
			plt.show() if show else plt.close()

		data = self._raw_data
		returns = self._returns

		def abs_oib_returns(tmp, ax):
			sns.scatterplot(data=tmp, x='Absolute Imbalance', y='Closing Return', ax=ax, hue='Volume quantile',
						 ec='k', palette='cubehelix_r')
			ax.set_title("Absolute order imbalance versus closing return by closing volume tercile")
			ax.set_xlabel("Absolute imbalance at auction start [million CHF]")
			xlimit = min(abs(min(tmp['Absolute Imbalance'])), max(tmp['Absolute Imbalance'])) * 1.4
			ax.set_xlim([-xlimit, xlimit])

		def rel_oib_returns(tmp, ax):  # Only marginally useful
			sns.scatterplot(data=tmp, x='Relative Imbalance', y='Closing Return', ax=ax, hue='Volume quantile',
						 ec='k', palette='cubehelix_r')
			ax.set_title("Relative order imbalance versus closing return by closing volume tercile")
			ax.set_xlabel("Relative order imbalance at auction start")
			ax.set_xlim([-1, 1])

		base_plot(data, abs_oib_returns, 'AbsoluteOIB_Returns')
		base_plot(data, rel_oib_returns, 'RelativeOIB_Returns')

	def plots_wpdc(self, save: bool = False, show: bool = True):
		lw = self._lw
		data = self._raw_data
		returns = self._returns
		wpdc_oib_df = data[['Relative Imbalance', 'Close turnover']].join(returns).reset_index()

		def base_plot(funcdata, axfunc, filename):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.grid(which='major', axis='both')
			ax.axhline(0, c='k', lw=1)
			axfunc(funcdata, ax)
			ax.set_ylabel('WPDC')
			ax.legend(ncol=2, fontsize='small')
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_axisbelow(True)
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\PriceDiscovery\\{1}".format(self._figdir, filename))
			plt.show() if show else plt.close()

		def wpdc_time_plot(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='WPDC', ax=ax, hue='Volume quantile', estimator='sum',
					   ci=None, lw=lw, palette='cubehelix_r', marker='.', mew=0, ms=9)
			ax.set_title("Weighted price discovery contribution per day over time in 2019")
			ax.set_xlabel("")
			ax.set_xlim([pd.datetime(2019, 1, 1), pd.datetime(2020, 1, 1)])
			locator = dates.MonthLocator()
			formatter = dates.ConciseDateFormatter(locator, show_offset=False)
			ax.xaxis.set_major_locator(locator)
			ax.xaxis.set_major_formatter(formatter)

		def wpdc_oib_stockday_plot(funcdata, ax):
			sns.scatterplot(data=funcdata, x='Relative Imbalance', y='WPDC', ax=ax, hue='Volume quantile',
						 ec='k', palette='cubehelix_r')
			ax.set_title("Weighted price discovery contribution and relative order imbalance by stockday")
			ax.set_xlabel("Relative imbalance")
			ax.axvline(0, c='k', lw=1)

		def wpdc_oib_stock_plot(funcdata, ax):
			funtmp = funcdata[funcdata['return_open_close'] != 0]
			funtmp = funtmp.groupby('Symbol').median().reset_index()
			sns.scatterplot(data=funtmp, x='Relative Imbalance', y='PDC', ax=ax, size='Symbol', hue='Symbol', ec='k',
						 palette='cubehelix', hue_order=self._avg_turnover.index,
						 sizes=(10, 400), size_order=self._avg_turnover.index)
			ax.set_title("Weighted price discovery contribution and relative order imbalance by stock")
			ax.set_xlabel("Relative imbalance")
			# ax.legend(loc='lower right')
			ax.set_xlim(right=0.6)
			ax.set_ylim(top=0.1, bottom=-0.1)
			ax.axvline(0, c='k', lw=1)

		def wpdc_oib_day_plot(funcdata, ax):
			funtmp = funcdata[funcdata['return_open_close'] != 0]
			funtmp.loc[:, 'Date'] = "-" + (funtmp['Date'].dt.dayofyear).astype(str)
			funtmp = funtmp.groupby('Date').mean().reset_index()
			funtmp.rename(columns={'Close turnover': 'Turnover [mn. CHF]'}, inplace=True)

			sns.scatterplot(data=funtmp, x='Relative Imbalance', y='WPDCday', ax=ax, size='Turnover [mn. CHF]',
						 ec='k', sizes=(5, 400), legend='brief', hue='Turnover [mn. CHF]', palette='cubehelix_r')
			ax.axvline(0, c='k', lw=1)
			ax.set_title("Weighted price discovery contribution and relative order imbalance by day")
			ax.set_xlabel("Relative Imbalance")
			ax.set_xlim([-.22, .22])
			ax.set_ylim([-.4, .4])

		base_plot(returns.reset_index(), wpdc_time_plot, 'WPDC_time')
		base_plot(wpdc_oib_df, wpdc_oib_stockday_plot, 'WPDC_OIB_stockday')
		base_plot(wpdc_oib_df, wpdc_oib_stock_plot, 'WPDC_OIB_stock')
		base_plot(wpdc_oib_df, wpdc_oib_day_plot, 'WPDC_OIB_day')

	def plot_stocks_time_compare(self, nstocks=6, save: bool = False, show: bool = True) -> None:
		lw = self._lw
		ms = self._ms
		stocks = self._avg_turnover.index[:nstocks].to_list()
		data = self._raw_data.reset_index(inplace=False)

		def base_plot(plotfunc, handle):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.axhline(y=0, lw=1, c='k')
			plotfunc(ax)
			ax.grid(which='major', axis='y')
			ax.set_xlabel("")
			handles, labels = ax.get_legend_handles_labels()
			ax.legend(handles=handles[1:], labels=labels[1:], fontsize='small', ncol=2)
			ax.set_xlim([pd.datetime(2019, 1, 1), pd.datetime(2020, 1, 1)])
			loca = dates.MonthLocator()
			form = dates.ConciseDateFormatter(loca, show_offset=False)
			ax.xaxis.set_major_locator(loca)
			ax.xaxis.set_major_formatter(form)
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\PriceDiscovery\\Stocks_{1}".format(self._figdir, handle))
			plt.show() if show else plt.close()

		def turnover_plot(ax):
			sns.lineplot(data=data, x='Date', y='Close turnover', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_title("Closing turnover over time for the {n} largest SLI titles".format(n=nstocks))
			ax.set_ylabel("Closing turnover [million CHF]")
			ax.set_ylim(bottom=0)

		def absolute_oib_plot(ax):
			sns.lineplot(data=data, x='Date', y='Absolute Imbalance', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_title("Absolute order imbalance over time for the {n} largest SLI titles".format(n=nstocks))
			ax.set_ylabel("Absolute order imbalance [million CHF]")

		def relative_oib_plot(ax):
			sns.lineplot(data=data, x='Date', y='Relative Imbalance', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_ylabel("Relative order imbalance")
			ax.set_title("Relative order imbalance over time for the {n} largest SLI titles".format(n=nstocks))

		def deviation_plot(ax):
			sns.lineplot(data=data, x='Date', y='Closing Return', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_title("Closing auction return over time for the {n} largest SLI titles".format(n=nstocks))
			ax.set_ylabel('Return during closing auction [\%]')
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_ylim([-.01, .01])

		base_plot(turnover_plot, 'Turnover')
		base_plot(absolute_oib_plot, 'AbsoluteImbalance')
		base_plot(relative_oib_plot, 'RelativeImbalance')
		base_plot(deviation_plot, 'Deviation')


class IntervalVisual(Visualization):
	def __init__(self, datapath):
		super().__init__(datapath)
		self._extract_indicators()
		self._avg_turnover = self._raw_data['Close turnover'].groupby('Symbol').mean().sort_values(ascending=False)

		# Attributes
		self._lw = 1.3
		self._ms = 9

	def _extract_indicators(self) -> None:
		data = self._raw_data.sort_index()
		data['Close turnover'] = data['close_price'] * data['close_vol'] / 10 ** 6
		data['Absolute Imbalance'] = (data['snap_bids'] - data['snap_asks']) * data['close_price'] / 10 ** 6
		data['Relative Imbalance'] = ((data['snap_bids'] - data['snap_asks']) / (data['snap_bids'] + data['snap_asks']))
		data['Turnover'] = data['snap_price'] * data['snap_vol'] / 10 ** 6
		data['Deviation'] = (data['snap_price'] - data['close_price']) / data['close_price'] * 10 ** 4
		data['Month'] = data['Date'].dt.strftime('%B')
		data.set_index(['Date', 'Symbol', 'Lag'], inplace=True)


		# Calculate WPDC
		data['snap_return'] = (data['snap_price'] / data['snap_price'].shift(periods=1)) - 1
		data.loc[data.index.get_level_values('Lag') == 0, 'snap_return'] = np.nan
		start, end = data['snap_price'].xs(0, level='Lag'), data['snap_price'].xs(600, level='Lag')
		close_df = pd.DataFrame({'close_return': (end / start) - 1})
		close_df = close_df.join(close_df['close_return'].abs().groupby('Date').sum(), rsuffix='_abs_total')
		data = data.join(close_df)

		data['PDC'] = data['snap_return'] / data['close_return']
		data['weights'] = abs(data['close_return'] / data['close_return_abs_total'])
		data['WPDC'] = data['weights'] * data['PDC']
		self._raw_data = data.join(data['WPDC'].groupby(['Date','Lag']).sum(), rsuffix='interval')

	def plot_months(self, save: bool = False, show: bool = True) -> None:
		lw = self._lw
		ms = self._ms
		raw = self._raw_data.copy(deep=True)
		tmp = raw.groupby(['Lag', 'Date']).mean().reset_index()
		tmp['Month'] = tmp['Date'].apply(lambda d: d.strftime('%B'))

		def base_plot(linefunc, filename):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.axhline(0, c='k', lw=1, ls='solid')
			linefunc(ax)
			ax.grid(which='major', axis='y')
			ax.set_xlabel("Seconds since start of auction")
			handles, labels = ax.get_legend_handles_labels()
			ax.legend(handles=handles[1:], labels=labels[1:], ncol=3, fontsize='small')
			ax.set_xlim([0, 600])
			ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\Intervals\\Monthly\\{1}".format(self._figdir, filename), transparent=True)
			plt.show() if show else plt.close()

		def volume_plot(ax):
			sns.lineplot(data=tmp, x='Lag', y='Turnover', hue='Month', ci=None,
					   palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Average hypothetical closing turnover per SLI title per day")
			ax.set_ylabel("Turnover [million CHF]")
			ax.set_ylim(bottom=0)

		def rel_oib_plot(ax):
			sns.lineplot(data=tmp, x='Lag', y='Relative Imbalance', hue='Month', ci=None,
					   palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Average relative order imbalance per SLI title per day")
			ax.set_ylabel("Relative order imbalance")
			ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

		def abs_oib_plot(ax):
			sns.lineplot(data=tmp, x='Lag', y='Absolute Imbalance', hue='Month', ci=None, estimator='sum',
					   palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Total market-wide absolute order imbalance per day")
			ax.set_ylabel("Order imbalance [million CHF]")

		def price_average_plot(ax):
			sns.lineplot(data=tmp, x='Lag', y='Deviation', hue='Month', ci=None,
					   palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Average price deviation per SLI title per day")
			ax.set_ylabel("Price deviation [bps]")

		def price_median_plot(ax):
			sns.lineplot(data=tmp, x='Lag', y='Deviation', hue='Month', ci=None, estimator='median',
					   palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Median price deviation per SLI title per day")
			ax.set_ylabel("Price deviation [bps]")

		def wpdc_plot(ax):
			fundata = tmp[tmp['close_return'] != 0].groupby(['Date','Lag']).first().reset_index()
			sns.lineplot(data=fundata, x='Lag', y='WPDCinterval', hue='Month', ci=None, estimator='mean',
					   palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Average WPDC per 30-second interval across SLI titles")
			ax.set_ylabel("WPDC")

		def median_wpdc_plot(ax):
			fundata = tmp[tmp['close_return'] != 0].groupby(['Date','Lag']).first().reset_index()
			sns.lineplot(data=fundata, x='Lag', y='WPDCinterval', hue='Month', ci=None, estimator='median',
					   palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Median WPDC per 30-second interval across SLI titles")
			ax.set_ylabel("WPDC")

		base_plot(volume_plot, 'VolumePlotMonthly')
		base_plot(rel_oib_plot, 'RelativeOIBMonthly')
		base_plot(abs_oib_plot, 'AbsoluteOIBMonthly')
		base_plot(price_average_plot, 'PriceDeviationMonthly')
		base_plot(price_median_plot, 'MedianPriceDeviationMonthly')
		base_plot(wpdc_plot, 'WPDCMonthly')
		base_plot(median_wpdc_plot, 'WPDCMedianMonthly')

	def plot_stocks_lags_compare(self, nst=8, save: bool = False, show: bool = True) -> None:
		lw = self._lw
		ms = self._ms
		stocks = self._avg_turnover.index[:nst].to_list()
		data = self._raw_data.reset_index(inplace=False)
		data = data[data['Symbol'].isin(stocks)]

		def base_plot(plotfunc, handle):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.axhline(y=0, lw=1, c='k', ls='solid')
			plotfunc(ax)
			ax.grid(which='major', axis='y')
			ax.set_xlabel("Seconds since start of auction")
			handles, labels = ax.get_legend_handles_labels()
			ax.legend(handles=handles[1:], labels=labels[1:], ncol=2, fontsize='small')
			ax.set_xlim([0, 600])
			ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\Intervals\\Comparison\\{1}".format(self._figdir, handle), transparent=True)
			plt.show() if show else plt.close()

		def turnover_plot(ax):
			sns.lineplot(data=data, x='Lag', y='Turnover', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_ylim(bottom=0)
			ax.set_ylabel("Turnover [million CHF]")
			ax.set_title("Average hypothetical turnover throughout closing auction for {n} largest SLI titles".format(n=nst))
			ax.set_ylim(bottom=0)

		def absolute_oib_plot(ax):
			sns.lineplot(data=data, x='Lag', y='Absolute Imbalance', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_title("Average absolute order imbalance throughout closing auction for {n} largest SLI titles".format(n=nst))
			ax.set_ylabel("Absolute imbalance [million CHF]")

		def relative_oib_plot(ax):
			ax.axhline(y=0, lw=1, c='k', ls='dashed')
			sns.lineplot(data=data, x='Lag', y='Relative Imbalance', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_title("Average relative order imbalance throughout closing auction for {n} largest SLI titles".format(n=nst))

		def deviation_plot(ax):
			sns.lineplot(data=data, x='Lag', y='Deviation', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_ylabel('Average deviation from closing price [bps]')
			ax.set_title("Average deviation from closing price throughout closing auction for {n} largest SLI titles".format(n=nst))

		def deviation_median_plot(ax):
			sns.lineplot(data=data, x='Lag', y='Deviation', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None, estimator='median')
			ax.set_ylabel('Median deviation from closing price [bps]')
			ax.set_title("Median deviation from closing price throughout closing auction for {n} largest SLI titles".format(n=nst))

		def wpdc_plot(ax):
			tmp_data = data[(data['close_return'] != 0) & (data['Symbol'] != 'CFR')]
			sns.lineplot(data=tmp_data, x='Lag', y='WPDC', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None, estimator='mean')
			ax.set_ylabel('WPDC')
			ax.set_title("Average WPDC throughout closing auction for {n} largest SLI titles".format(n=nst))

		def median_wpdc_plot(ax):
			tmp_data = data[(data['close_return'] != 0) & (data['Symbol'] != 'CFR')]
			sns.lineplot(data=tmp_data, x='Lag', y='WPDC', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
					   ax=ax, palette='cubehelix', hue_order=stocks, ci=None, estimator='median')
			ax.set_ylabel('WPDC')
			ax.set_title("Median WPDC throughout closing auction for {n} largest SLI titles".format(n=nst))

		base_plot(turnover_plot, 'Turnover')
		base_plot(absolute_oib_plot, 'AbsoluteTurnover')
		base_plot(relative_oib_plot, 'RelativeTurnover')
		base_plot(deviation_plot, 'AverageDeviation')
		base_plot(deviation_median_plot, 'MedianDeviation')
		base_plot(wpdc_plot, 'WPDC')
		base_plot(median_wpdc_plot, 'WPDCMedian')

	def plot_stocks_within(self, nstocks=5, save: bool = False, show: bool = True) -> None:
		"""
		Very chaotic representation, but it indicates that many of the one-sided order imbalances are persistent.
		This probably comes from large rebalancing.
		"""
		dpi = 300
		def base_plot(funcdata, plotfunc, stock, handle):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=dpi)
			ax.axhline(0, c='k', lw=1.2)
			plotfunc(funcdata, ax, stock)
			ax.grid(which='major', axis='both')
			ax.set_xlabel("Seconds since start of auction")
			handles, labels = ax.get_legend_handles_labels()
			ax.legend(handles=handles[1:], labels=labels[1:], fontsize='small', ncol=6)
			ax.set_xlim([0, 600])
			ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\Intervals\\Within\\{1}_{2}".format(self._figdir, handle, stock), transparent=True)
			plt.show() if show else plt.close()

		stocks = self._avg_turnover.index[:nstocks].to_list()
		data = self._raw_data.reset_index(inplace=False)

		def deviation_plot(data, ax, stock):
			sns.lineplot(data=data, x='Lag', y='Deviation', ax=ax, hue='Month', legend='brief',
					   estimator=None, lw=0.5, palette='rainbow', units='Date')
			ax.set_ylim([-500, 500])
			ax.set_ylabel("Deviation from closing price [bps]k")
			ax.set_title("{0}: Deviation from closing price throughout auction (each line represents one trading day)".format(stock))

		def relative_oib_plot(data, ax, stock):
			sns.lineplot(data=data, x='Lag', y='Relative Imbalance', ax=ax, units='Date',
					   estimator=None, lw=0.5, palette='rainbow', hue='Month')
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_title("{0}: Relative order imbalance throughout auction (each line represents one trading day)".format(stock))
			ax.set_ylabel("Relative Imbalance [\%]")

		def absolute_oib_plot(data, ax, stock):
			sns.lineplot(data=data, x='Lag', y='Absolute Imbalance', ax=ax, hue='Month',
					   estimator=None, lw=0.5, palette='rainbow', units='Date')
			ax.set_title("{0}: Absolute order imbalance throughout auction (each line represents one trading day)".format(stock))
			ax.set_ylabel("Absolute Imbalance [million CHF]")

		def wpdc_plot(data, ax, stock):
			funcdata = data[data['close_return'] != 0]
			sns.lineplot(data=funcdata, x='Lag', y='WPDC', ax=ax, hue='Month',
					   estimator=None, lw=0.5, palette='rainbow', units='Date')
			ax.set_title("{0}: WPDC throughout auction (each line represents one trading day)".format(stock))
			ax.set_ylabel("WPDC")

		for s in stocks:
			tmp = data[data['Symbol'] == s].sort_values('Date', ascending=True)
			base_plot(tmp, deviation_plot, s, 'Deviation')
			base_plot(tmp, relative_oib_plot, s, 'RelativeImbalance')
			base_plot(tmp, absolute_oib_plot, s, 'AbsoluteImbalance')
			base_plot(tmp, wpdc_plot, s, 'WPDC')
