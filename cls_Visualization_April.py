import itertools
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import dates
from matplotlib import pyplot as plt
from matplotlib import ticker
from pandas.plotting import register_matplotlib_converters

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

register_matplotlib_converters()

pd.set_option('display.width', 300)
pd.set_option("display.max_columns", 16)
def_palette = "Set1"
sns.set_palette(def_palette, desat=0.8)
def_color = sns.color_palette(def_palette, 1)[0]


class Visualization:
	_panel2 = (18 / 2.5, 20 / 2.5)
	_panel3 = (18 / 2.5, 24 / 2.5)
	
	def __init__(self, datapath):
		self._raw_data = pd.read_csv(datapath, parse_dates=['Date'])
		self._figsize = (22 / 2.54, 13 / 2.54)
		self._cwd = os.getcwd() + "\\03 Presentation April"
		self._figdir = self._cwd + "\\Figures"
		self._dpi = 450
		
		print("--- SuperClass Initiated ---")
		
	def return_data(self):
		return self._raw_data


class SensVisual(Visualization):
	def __init__(self, datapath: str, base: str, limit: float = 0.3):
		super().__init__(datapath)
		self._manipulate_data()
		self._raw_data.set_index(['Mode', 'Date', 'Symbol', 'Percent'], inplace=True)
		# self._raw_data.drop(index=['ALC'], level='Symbol', inplace=True)
		self._define_quantiles()
		
		# Attributes
		if base in {'SeparateOrders', 'FullLiquidity', 'CrossedVolume'}:
			self._base = base  # LimitOrders vs TotalVolume
		else:
			raise ValueError("base not in {'SeparateOrders','FullLiquidity','CrossedVolume'}")
		
		self._base_d = dict(SeparateOrders='Separate liquidity basis',
		                    FullLiquidity='Full liquidity basis',
		                    CrossedVolume='Execution volume basis')
		
		self._lmt_modes = dict(bid_limit="bid orders", ask_limit="ask orders", all_limit="bid/ask orders")
		self._mkt_modes = dict(bid_market="bid market", ask_market="ask market orders", all_market="bid/ask market orders",
		                       bid_cont="bid market orders (cont. phase)", ask_cont="ask market orders (cont. phase)",
		                       all_cont="bid/ask market orders (cont. phase)")
		self._limit = limit
	
	def _manipulate_data(self):
		data = self._raw_data
		data['Percent'] = data['Percent'].round(2)
		data['Closing turnover'] = data['close_price'] * data['close_vol']
		data['Deviation price'] = abs(data['adj_price'] - data['close_price']) / data['close_price'] * 10000
		data['Deviation turnover'] = (data['adj_vol'] - data['close_vol']) / data['close_vol']
		
		# Absolute deviations for symmetric removals
		sel = data.loc[data['Mode'].isin(['all_limit', 'all_market', 'all_cont']), 'Deviation price']
		data.loc[data['Mode'].isin(['all_limit', 'all_market', 'all_cont']), 'Deviation price'] = sel.abs()
		
		self._avg_turnover = data.groupby('Symbol')['Closing turnover'].mean().sort_values(ascending=False).dropna()
	
	def _define_quantiles(self):
		data = self._raw_data.sort_index()
		quants = data['Closing turnover'].groupby(['Mode', 'Date', 'Symbol']).first()
		quants = quants.groupby('Date').transform(
			lambda x: pd.qcut(x, q=3, labels=['least liquid', 'neutral', 'most liquid']).astype(str))
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
				if mode in self._mkt_modes.keys():
					plt.savefig(self._figdir + "\\Sensitivity\\MarketLiquidity\\remove_{n}_{m}".format(n=filename, m=mode),
					            transparent=True)
				else:
					plt.savefig(self._figdir + "\\Sensitivity\\{b}\\remove_{n}_{m}".format(n=filename, m=mode, b=b),
					            transparent=True)
			plt.show() if show else plt.close()
		
		def agg_price_fun(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='Deviation price', hue='Percent', estimator='mean',
			             ax=ax, lw=0.9, ci=None, err_kws=dict(alpha=0.2), palette='Reds_d')
			ax.set_ylabel("Deviation of closing price [bps]")
			ax.set_title("[{b}] Average effect of removal of {m} on closing price".format(b=self._base_d[b], m=self._lmt_modes[mode]))
		
		def agg_volume_fun(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='Deviation turnover', hue='Percent', estimator='mean',
			             ax=ax, lw=0.9, ci=None, err_kws=dict(alpha=0.2), palette='Blues_d')
			ax.set_ylabel("Deviation of closing volume [\%]")
			ax.set_title("[{b}] Average effect of removal of {m} on closing volume".format(b=self._base_d[b], m=self._lmt_modes[mode]))
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
			ax.set_ylim(top=0, bottom=-1)
		
		def agg_price_mkt_fun(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='Deviation price', hue='Volume quantile', estimator='mean',
			             ax=ax, lw=1.3, ci=None, err_kws=dict(alpha=0.2), palette='Reds')
			ax.set_ylabel("Deviation of closing price [bps]")
			ax.set_title("Average effect of removal of {m} on closing price".format(m=self._mkt_modes[mode]))
		
		def agg_volume_mkt_fun(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='Deviation turnover', hue='Volume quantile', estimator='mean',
			             ax=ax, lw=1.3, ci=None, err_kws=dict(alpha=0.2), palette='Blues')
			ax.set_ylabel("Deviation of closing volume [\%]")
			ax.set_title("Average effect of removal of {m} on closing volume".format(m=self._mkt_modes[mode]))
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_ylim(top=0, bottom=-1)
		
		# EXECUTION
		for mode in iter(self._lmt_modes.keys()):
			df = data.loc[mode, ['Deviation price', 'Deviation turnover']].reset_index(inplace=False)
			df = df[((df['Percent'] <= limit) & (df['Percent'] > 0)) | (df['Percent'] == 1)]
			df['Percent'] = (df['Percent'] * 100).astype(int).astype(str) + "\%"
			
			base_plot(df, agg_price_fun, 'Price')
			base_plot(df, agg_volume_fun, 'Turnover')
		
		for mode in iter(self._mkt_modes.keys()):
			df = data.loc[mode, ['Deviation price', 'Deviation turnover', 'Volume quantile']].reset_index(inplace=False)
			base_plot(df, agg_price_mkt_fun, 'Price')
			base_plot(df, agg_volume_mkt_fun, 'Turnover')
	
	def plot_removal_quantiles(self, save: bool = False, show: bool = True) -> None:
		limit = 0.35
		vol_lim = 0.8
		b = self._base
		data = self._raw_data[['Deviation price', 'Deviation turnover', 'Volume quantile']]
		
		def base_plot(funcdata, linefunc, label):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.grid(which='major', axis='y')
			ax.axhline(0, c='k', lw=1)
			linefunc(funcdata, ax)
			ax.legend(fontsize='small')
			ax.set_axisbelow(True)
			fig.tight_layout()
			if save:
				if 'market' in mode:
					plt.savefig(self._figdir + "\\Sensitivity\\MarketLiquidity\\quantile_removal_{n}".format(n=label),
					            transparent=True)
				else:
					plt.savefig(self._figdir + "\\Sensitivity\\{b}\\quantile_removal_{n}_{m}".format(n=label, m=mode, b=b),
					            transparent=True)
			plt.show() if show else plt.close()
		
		def quant_price_func(funcdata, ax):
			sns.boxplot(data=funcdata[funcdata['Percent'] <= limit], ax=ax, x='PercentStr', y='Deviation price',
			            hue='Volume quantile', showfliers=False, whis=[2.5, 97.5], palette='Reds', linewidth=1)
			ax.set_ylabel("Price deviation [bps]")
			ax.set_title("[{b}] Quantile distribution of changes in closing price by removing {m}".format(b=self._base_d[b],
			                                                                                              m=self._lmt_modes[mode]))
			ax.set_xlabel("Removed Liquidity [\%]")
		
		def quant_vol_func(funcdata, ax):
			sns.boxplot(data=funcdata[funcdata['Percent'] <= vol_lim], ax=ax, x='PercentStr', y='Deviation turnover',
			            hue='Volume quantile', showfliers=False, whis=[2.5, 97.5], palette='Blues', linewidth=1)
			ax.set_ylabel("Turnover deviation [\%]")
			ax.set_title("[{b}] Quantile distrubution of changes in closing turnover by removing {m}".format(b=self._base_d[b],
			                                                                                                 m=self._lmt_modes[mode]))
			ax.set_xlabel("Removed Liquidity [\%]")
			ax.set_ylim([-1.05, 0.05])
			ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		
		def quant_price_mkt_func(funcdata, ax):
			tmp = funcdata.reset_index()
			tmp.sort_values('Mode', inplace=True)
			tmp.replace(self._mkt_modes, inplace=True)
			sns.boxplot(data=tmp, ax=ax, x='Mode', y='Deviation price',
			            hue='Volume quantile', showfliers=False, whis=[2.5, 97.5], palette='Reds', linewidth=1)
			ax.set_ylabel("Price deviation [bps]")
			ax.set_xlabel("")
			ax.set_title("Quantile distribution of changes in closing price by removing {m}".format(m=mode))
			ax.set_ylim([-300, 250])
			ax.yaxis.set_major_locator(ticker.MultipleLocator(base=25))
		
		def quant_vol_mkt_func(funcdata, ax):
			tmp = funcdata.reset_index()
			tmp.sort_values('Mode', inplace=True)
			tmp.replace(self._mkt_modes, inplace=True)
			sns.boxplot(data=tmp, ax=ax, x='Mode', y='Deviation turnover', hue='Volume quantile',
			            showfliers=False, whis=[2.5, 97.5], palette='Blues', linewidth=1)
			ax.set_ylabel("Turnover deviation [\%]")
			ax.set_title("Quantile distrubution of changes in closing turnover by removing {m}".format(m=mode))
			ax.set_xlabel("")
			ax.set_ylim([-1.05, 0.05])
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.1))
		
		for mode in iter(self._lmt_modes.keys()):
			tmp = data.loc[mode, :].reset_index(inplace=False)
			tmp = tmp[tmp['Percent'] > 0]
			tmp['PercentStr'] = (tmp['Percent'] * 100).astype(int).astype(str) + '\%'
			base_plot(tmp, quant_price_func, 'Price')
			base_plot(tmp, quant_vol_func, 'Turnover')
		
		mode = 'market orders'
		base_plot(data.loc[['bid_market', 'ask_market', 'all_market'], :], quant_price_mkt_func, 'market_price')
		base_plot(data.loc[['bid_market', 'ask_market', 'all_market'], :], quant_vol_mkt_func, 'market_volume')
		mode = 'market orders from continuous phase'
		base_plot(data.loc[['bid_cont', 'ask_cont', 'all_cont'], :], quant_price_mkt_func, 'cont_price')
		base_plot(data.loc[['bid_cont', 'ask_cont', 'all_cont'], :], quant_vol_mkt_func, 'cont_volume')
	
	def plot_removal_by_stock(self, stock: str, save: bool = False, show: bool = False):
		limit = 0.35
		vol_lim = 0.8
		b = self._base
		data = self._raw_data[['Deviation price', 'Deviation turnover']]
		symbols = self._avg_turnover[:15].index
		
		def base_plot(funcdata, linefunc, label=None):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.grid(which='major', axis='y')
			ax.axhline(0, c='k', lw=1)
			linefunc(funcdata, ax)
			ax.legend(fontsize='small')
			ax.set_ylim(bottom=0)
			ax.set_axisbelow(True)
			plt.xticks(rotation='vertical')
			fig.tight_layout()
			# if save:
			# 	if 'market' in mode:
			# 		plt.savefig(self._figdir + "\\Sensitivity\\MarketLiquidity\\quantile_removal_{n}".format(n=label),
			# 		            transparent=True)
			# 	else:
			# 		pass
			# 		plt.savefig(self._figdir + "\\Sensitivity\\{b}\\quantile_removal_{n}_{m}".format(n=label, b=b),
			# 		            transparent=True)
			plt.show() if show else plt.close()
		
		def stockplot_box(funcdata, ax):
			tmp = funcdata[funcdata['Symbol'].isin(symbols)]
			sns.boxplot(data=tmp[tmp['Percent'] == 0.1], ax=ax, x='Symbol', y='Deviation price',
			            hue='Mode', showfliers=False, whis=[2.5, 97.5], palette='cubehelix', linewidth=0.8, order=symbols)
			ax.set_ylabel("Price deviation [bps]")
			ax.set_title(
				"[{b}]: Effect of removing 10\% of liquidity by stock (+/- 95\% of observations)".format(b=self._base_d[b], s=stock))
			ax.set_xlabel("")
		
		def stockplot_cross(funcdata, ax):
			tmp = funcdata[funcdata['Symbol'].isin(symbols)]
			sns.barplot(data=tmp[tmp['Percent'] == 0.1],
			            ax=ax, x='Symbol', y='Deviation price', order=symbols,
			            hue='Mode', capsize=0.2, palette='cubehelix', ci='sd', errwidth=0.8, dodge=True)
			ax.set_ylabel("Price deviation [bps]")
			ax.set_title(
				"[{b}]: Effect of removing 10\% of liquidity by stock (+/- 1 standard deviation)".format(b=self._base_d[b], s=stock))
			ax.set_xlabel("")
		
		df = data.loc[['bid_limit', 'ask_limit'], :].reset_index()
		df = df[df['Percent'] > 0]
		df['Deviation price'] = df['Deviation price'].abs()
		df['PercentStr'] = (df['Percent'] * 100).astype(int).astype(str) + '\%'
		df.replace(self._lmt_modes, inplace=True)
		base_plot(df, stockplot_box)
		base_plot(df, stockplot_cross)
	
	def plots_report(self, save: bool = False, show: bool = True) -> None:
		figdir = f"{os.getcwd()}\\06 Figures\\"
		limit = 0.35
		vol_lim = 0.8
		b = self._base
		data = self._raw_data[['Deviation price', 'Deviation turnover', 'Volume quantile']]
		
		def limit_dataprep(mode):
			tmp = data.loc[mode, :].reset_index(inplace=False)
			tmp = tmp[tmp['Percent'] > 0]
			tmp['Deviation price'] = tmp['Deviation price'].abs()
			tmp['PercentStr'] = (tmp['Percent'] * 100).astype(int).astype(str) + '\%'
			return tmp
		
		def quant_price_func(funcdata, ax):
			sns.boxplot(data=funcdata[funcdata['Percent'] <= limit], ax=ax, x='PercentStr', y='Deviation price',
			            hue='Volume quantile', showfliers=False, whis=[2.5, 97.5], palette='cubehelix', linewidth=1)
			ax.set_ylabel("Absolute price deviation [bps]")
			ax.grid(which='major', axis='y')
			ax.axhline(0, c='k', lw=1)
			ax.legend(fontsize='small', loc='upper left')
			ax.set_axisbelow(True)
			
			ax.set_xlabel("Removed Liquidity [\%]")
		
		def quant_price_mkt_func(funcdata, ax):
			tmp = funcdata.reset_index()
			# tmp.sort_values('Mode', inplace=True)
			tmp.replace(self._mkt_modes, inplace=True)
			sns.boxplot(data=tmp, ax=ax, x='Mode', y='Deviation price',
			            hue='Volume quantile', showfliers=False, whis=[2.5, 97.5], palette='cubehelix', linewidth=1)
			ax.grid(which='major', axis='y')
			ax.axhline(0, c='k', lw=1)
			ax.set_axisbelow(True)
			ax.legend(fontsize='small', loc='upper left')
			ax.set_ylabel("Price deviation [bps]")
			ax.set_xlabel("")
		
		# LIMIT ORDER PLOTS
		if b in {'FullLiquidity', 'CrossedVolume'}:
			fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self._panel2)
			ax1.get_shared_y_axes().join(ax1, ax2)
			quant_price_func(limit_dataprep('bid_limit'), ax1)
			ax1.set_title("{Panel A}: Removal of bid limit orders")
			quant_price_func(limit_dataprep('ask_limit'), ax2)
			ax2.set_title("{Panel B}: Removal of ask limit orders")
		else:
			fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self._panel3)
			ax1.get_shared_y_axes().join(ax1, ax2)
			quant_price_func(limit_dataprep('bid_limit'), ax1)
			ax1.set_title("{Panel A}: Removal of bid limit orders")
			quant_price_func(limit_dataprep('ask_limit'), ax2)
			ax2.set_title("{Panel B}: Removal of ask limit orders")
			quant_price_func(limit_dataprep('all_limit'), ax3)
			ax3.set_title("{Panel C}: Removal of bid and ask limit orders")
		
		fig.tight_layout()
		if save:
			plt.savefig(f"{figdir}\\Sens_limit_{b}.pdf")
		plt.show() if show else plt.close()
		
		# MARKET ORDER PLOTS
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self._panel2)
		quant_price_mkt_func(data.loc[['bid_market', 'ask_market', 'all_market'], :], ax1)
		ax1.set_title("{Panel A}: Removal of all market orders")
		quant_price_mkt_func(data.loc[['bid_cont', 'ask_cont', 'all_cont'], :], ax2)
		ax2.set_title("{Panel B}: Removal of all market orders from continuous phase")
		
		fig.tight_layout()
		if save:
			plt.savefig(f"{figdir}\\Sens_mkt.pdf")
		plt.show() if show else plt.close()


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
		self._color_dict = {'least liquid': 2, 'neutral': 1, 'most liquid': 0}
		self._avg_turnover = self._raw_data['Close turnover'].groupby('Symbol').mean().sort_values(ascending=False)
	
	def _calculate_factors(self):
		data = self._raw_data
		rets = self._returns
		
		data['Close turnover'] = data['actual_close_price'] * data['close_vol'] / 10 ** 6
		data['Closing Return'] = (data['actual_close_price'] - data['pre_midquote']) / data['pre_midquote'] * 10 ** 4
		data['Volume Imbalance'] = (data['start_bids'] - data['start_asks']) * data['close_price'] / 10 ** 6
		data['Relative Imbalance'] = (data['start_bids'] - data['start_asks']) / (data['start_bids'] + data['start_asks'])
		
		# Calculate WPDC
		rets = rets.join(rets['return_open_close'].abs().groupby('Date').sum(), rsuffix='_abs_total', on='Date')
		rets['PDC'] = rets['return_close'] / rets['return_open_close']
		rets = rets[rets['return_open_close'] != 0]
		rets['TPDC'] = rets['PDC'].groupby('Symbol').transform(lambda x: np.mean(x) / (np.std(x) / np.sqrt(len(x))))
		rets['WPDC'] = rets['PDC'] * abs(rets['return_open_close'] / rets['return_open_close_abs_total'])
		self._returns = rets.join(rets['WPDC'].groupby('Date').sum(), rsuffix='day', on='Date', how='inner')
	
	def _define_quantiles(self):
		data = self._raw_data.sort_index()
		rets = self._returns.sort_index()
		
		quants = data['Close turnover'].groupby('Date').transform(
			lambda x: pd.qcut(x, 3, labels=['least liquid', 'neutral', 'most liquid']).astype(str))
		quants.rename('Volume quantile', inplace=True)
		self._raw_data = data.join(quants)
		self._returns = rets.join(quants)
	
	def plot_oib_returns(self, save: bool = False, show: bool = True) -> None:
		data = self._raw_data
		returns = self._returns
		wpdc_oib_df = data[['Relative Imbalance', 'Volume Imbalance', 'Closing Return', 'Close turnover']].join(returns).reset_index()
		cd = self._color_dict
		
		def base_plot(funcdata, linefunc, name):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.axvline(0, c='k', lw=1, zorder=1)
			ax.axhline(0, c='k', lw=1, zorder=1)
			
			linefunc(funcdata, ax)
			ax.set_axisbelow(True)
			ax.grid(which='major', axis='both')
			if linefunc != wpdc_oib_stockday_plot:
				ax.set_ylabel('Closing return [bps]')
				ax.set_ylim([-250, 250])
				ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
			# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			fig.tight_layout()
			if save:
				plt.savefig(self._figdir + "\\PriceDiscovery\\{n}_{m}".format(n=name, m=mode), transparent=True)
			plt.show() if show else plt.close()
		
		def volume_oib_returns(funcdata, ax):
			if mode == 'all':
				sns.scatterplot(data=funcdata, x='Volume Imbalance', y='Closing Return', ax=ax, hue='Volume quantile',
				                ec='k', palette='cubehelix_r', zorder=2)
			else:
				sns.scatterplot(data=funcdata[funcdata['Volume quantile'] == mode], x='Volume Imbalance', y='Closing Return',
				                ax=ax, ec='k', color=sns.color_palette('cubehelix', n_colors=3)[cd[mode]], label=mode, zorder=2)
			ax.set_title("Volume order imbalance versus closing return by closing volume tercile ({m})".format(m=mode))
			ax.set_xlabel("Volume imbalance at auction start [million CHF]")
			ax.xaxis.set_major_locator(ticker.MultipleLocator(base=25))
			ax.set_xlim([-150, 150])
		
		def relative_oib_returns(funcdata, ax):
			if mode == 'all':
				sns.scatterplot(data=funcdata, x='Relative Imbalance', y='Closing Return', ax=ax, hue='Volume quantile',
				                ec='k', palette='cubehelix_r', zorder=2)
			else:
				sns.scatterplot(data=funcdata[funcdata['Volume quantile'] == mode], x='Relative Imbalance', y='Closing Return',
				                ax=ax, ec='k', color=sns.color_palette('cubehelix', n_colors=3)[cd[mode]], label=mode, zorder=2)
			ax.set_title("Relative order imbalance versus closing return by closing volume tercile ({m})".format(m=mode))
			ax.set_xlabel("Relative order imbalance at auction start")
			ax.set_xlim([-1, 1])
		
		def wpdc_oib_stockday_plot(funcdata, ax):
			if mode == 'all':
				sns.scatterplot(data=funcdata, x='Relative Imbalance', y='WPDC', ax=ax, hue='Volume quantile',
				                ec='k', palette='cubehelix_r', zorder=2)
			else:
				sns.scatterplot(data=funcdata[funcdata['Volume quantile'] == mode], x='Relative Imbalance', y='WPDC', ax=ax,
				                ec='k', zorder=2, color=sns.color_palette('cubehelix', n_colors=3)[cd[mode]], label=mode)
			ax.set_title("Weighted price discovery contribution and relative order imbalance by stock-day ({m})".format(m=mode))
			ax.set_xlabel("Relative imbalance")
			ax.set_ylabel("$WPDC_{d,s,i}$")
			ax.set_xlim([-1, 1])
			ax.set_ylim([-0.15, 0.15])
		
		for mode in {'most liquid', 'neutral', 'least liquid', 'all'}:
			base_plot(wpdc_oib_df, volume_oib_returns, 'VolumeOIB_Returns')
			base_plot(wpdc_oib_df, relative_oib_returns, 'RelativeOIB_Returns')
			base_plot(wpdc_oib_df, wpdc_oib_stockday_plot, 'WPDCStockdayImbalance')
	
	def plots_wpdc(self, save: bool = False, show: bool = True):
		lw = self._lw
		data = self._raw_data
		returns = self._returns
		wpdc_oib_df = data[['Relative Imbalance', 'Close turnover']].join(returns).reset_index()
		
		def base_plot(funcdata, axfunc, filename):
			fig, ax = plt.subplots(1, 1, figsize=self._figsize, dpi=self._dpi)
			ax.grid(which='major', axis='both')
			ax.axhline(0, c='k', lw=1, zorder=1)
			axfunc(funcdata, ax)
			ax.legend(ncol=2, fontsize='small', loc='lower right')
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_axisbelow(True)
			fig.tight_layout()
			if save:
				plt.savefig("{0}\\PriceDiscovery\\{1}".format(self._figdir, filename), transparent=True)
			plt.show() if show else plt.close()
		
		def wpdc_time_plot(funcdata, ax):
			sns.lineplot(data=funcdata, x='Date', y='WPDC', ax=ax, hue='Volume quantile', estimator='sum',
			             ci=None, lw=lw, palette='cubehelix_r', marker='.', mew=0, ms=9, zorder=2)
			ax.set_title("Weighted price discovery contribution per day over time in 2019")
			ax.set_xlabel("")
			ax.set_ylabel("$WPDC_{d,i}$")
			ax.set_xlim([pd.datetime(2019, 1, 1), pd.datetime(2020, 1, 1)])
			locator = dates.MonthLocator()
			formatter = dates.ConciseDateFormatter(locator, show_offset=False)
			ax.xaxis.set_major_locator(locator)
			ax.xaxis.set_major_formatter(formatter)
		
		def wpdc_oib_stock_plot(funcdata, ax):
			funtmp = funcdata[funcdata['return_open_close'] != 0]
			funtmp = funtmp.groupby('Symbol').median().reset_index()
			sns.scatterplot(data=funtmp, x='Relative Imbalance', y='PDC', ax=ax, size='Symbol', hue='Symbol', ec='k',
			                palette='cubehelix', hue_order=self._avg_turnover.index,
			                sizes=(10, 400), size_order=self._avg_turnover.index, zorder=2)
			ax.set_title("Weighted price discovery contribution and relative order imbalance by stock")
			ax.set_xlabel("Relative imbalance")
			ax.set_ylabel("$WPDC_{s,i}$")
			ax.set_xlim(right=0.6)
			ax.set_ylim(top=0.1, bottom=-0.1)
			ax.axvline(0, c='k', lw=1)
		
		def wpdc_oib_day_plot(funcdata, ax):
			funtmp = funcdata[funcdata['return_open_close'] != 0]
			funtmp.loc[:, 'Date'] = "-" + (funtmp['Date'].dt.dayofyear).astype(str)
			funtmp = funtmp.groupby('Date').mean().reset_index()
			funtmp.rename(columns={'Close turnover': 'Turnover [mn. CHF]'}, inplace=True)
			
			sns.scatterplot(data=funtmp, x='Relative Imbalance', y='WPDCday', ax=ax, size='Turnover [mn. CHF]', ec='k',
			                sizes=(5, 400), legend='brief', hue='Turnover [mn. CHF]', palette='cubehelix_r', zorder=2)
			ax.axvline(0, c='k', lw=1)
			ax.set_title("Weighted price discovery contribution and relative order imbalance by day")
			ax.set_xlabel("Relative Imbalance")
			ax.set_ylabel("$WPDC_{d,i}$")
			ax.set_xlim([-.22, .22])
			ax.set_ylim([-.4, .4])
		
		base_plot(returns.reset_index(), wpdc_time_plot, 'WPDC_time')
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
				plt.savefig("{0}\\PriceDiscovery\\Stocks_{1}".format(self._figdir, handle), transparent=True)
			plt.show() if show else plt.close()
		
		def turnover_plot(ax):
			sns.lineplot(data=data, x='Date', y='Close turnover', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
			             ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_title("Closing turnover over time for the {n} largest SLI titles".format(n=nstocks))
			ax.set_ylabel("Closing turnover [million CHF]")
			ax.set_ylim(bottom=0)
		
		def volume_oib_plot(ax):
			sns.lineplot(data=data, x='Date', y='Volume Imbalance', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
			             ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_title("Volume order imbalance over time for the {n} largest SLI titles".format(n=nstocks))
			ax.set_ylabel("Volume order imbalance [million CHF]")
		
		def relative_oib_plot(ax):
			sns.lineplot(data=data, x='Date', y='Relative Imbalance', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
			             ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_ylabel("Relative order imbalance")
			ax.set_title("Relative order imbalance over time for the {n} largest SLI titles".format(n=nstocks))
		
		def deviation_plot(ax):
			sns.lineplot(data=data, x='Date', y='Closing Return', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
			             ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_title("Closing auction return over time for the {n} largest SLI titles".format(n=nstocks))
			ax.set_ylabel('Return during closing auction [bps]')
			# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_ylim([-100, 100])
		
		base_plot(turnover_plot, 'Turnover')
		base_plot(volume_oib_plot, 'AbsoluteImbalance')
		base_plot(relative_oib_plot, 'RelativeImbalance')
		base_plot(deviation_plot, 'Deviation')
	
	def plots_report(self, save: bool = False, show: bool = True):
		figdir = f"{os.getcwd()}\\06 Figures"
		data = self._raw_data
		returns = self._returns
		wpdc_oib_df = data[['Relative Imbalance', 'Close turnover']].join(returns).reset_index()
		
		def wpdc_oib_all_plot(funcdata, ax):
			tmp = funcdata[funcdata['return_open_close'] != 0].reset_index(inplace=False)
			# tmp['Close turnover'] = np.log10(tmp['Close turnover'])
			tmp = tmp.sort_values('Close turnover', ascending=True)
			sns.scatterplot(data=tmp, x='Relative Imbalance', y='WPDC', ax=ax, size='Close turnover',
			                hue='Close turnover', ec='k', palette='cubehelix_r', sizes=(10, 400), zorder=2)
			# sns.scatterplot(data=tmp, x='Relative Imbalance', y='WPDC', ax=ax, hue='Close turnover', ec='k',
			#                 palette='cubehelix_r', zorder=2)
			ax.set_title(r"Panel A: $WPDC^{{CL}}_{{d,s}}$ and order imbalance ($N = D \times S = 7470$)")
			ax.set_xlabel("$IMBAL$")
			ax.set_ylabel("$WPDC^{CL}_{d,s}$")
			ax.set_ylim((-0.1, 0.1))
			ax.set_xlim((-0.8, 0.8))
			ax.grid(which='major', axis='both')
			ax.axvline(0, c='k', lw=1, zorder=1)
			ax.axhline(0, c='k', lw=1, zorder=1)
			ax.set_axisbelow(True)
			ax.legend(fontsize='small', loc='lower right', title='Turnover [mn. CHF]', ncol=2)
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
	
		def wpdc_oib_day_plot(funcdata, ax):
			tmp = funcdata[funcdata['return_open_close'] != 0]
			# tmp.loc[:, 'Date'] = "-" + (tmp['Date'].dt.dayofyear).astype(str)
			tmp = tmp.groupby('Date').agg({'Close turnover': sum, 'WPDC': sum, 'Relative Imbalance': np.mean}).reset_index()
			tmp.sort_values('Close turnover', ascending=True, inplace=True)
			
			sns.scatterplot(data=tmp, x='Relative Imbalance', y='WPDC', ax=ax, size='Close turnover', ec='k',
			                sizes=(5, 500), legend='brief', hue='Close turnover', palette='cubehelix_r', zorder=2)
			ax.set_ylim(bottom=-0.3, top=0.3)
			ax.set_xlim((-0.25, 0.25))
			ax.axvline(0, c='k', lw=1, zorder=1)
			ax.axhline(0, c='k', lw=1, zorder=1)
			ax.grid(which='major', axis='both')
			ax.set_title(f"Panel B: $WPDC^{{CL}}_{{d}}$ and average order imbalance by day ($D = {tmp.shape[0]}$)")
			ax.set_xlabel("$IMBAL$")
			ax.set_ylabel("$WPDC^{CL}_{d}$")
			ax.set_axisbelow(True)
			ax.legend(loc='lower right', title='Turnover [mn. CHF]', ncol=2, fontsize='small')
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
		
		def wpdc_oib_stock_plot(funcdata, ax):
			tmp = funcdata[funcdata['return_open_close'] != 0]
			tmp = tmp.groupby('Symbol').mean().reset_index()
			sns.scatterplot(data=tmp, x='Relative Imbalance', y='TPDC', ax=ax, size='Symbol', hue='Symbol', ec='k', palette='cubehelix',
			                hue_order=self._avg_turnover.index, sizes=(10, 500), size_order=self._avg_turnover.index, zorder=2)
			ax.set_title(f"Panel C: $TPDC^{{CL}}_{{s}}$ and average order imbalance by stock ($S = {tmp.shape[0]}$)")
			ax.axhspan(-1.96, 1.96, alpha=0.2, color='grey')
			ax.set_xlabel("$IMBAL$")
			ax.set_ylabel("$TPDC^{CL}_{s}$")
			ax.set_ylim((-3, 3))
			ax.set_xlim((-0.6, 0.6))
			ax.grid(which='major', axis='both')
			ax.axvline(0, c='k', lw=1, zorder=1)
			ax.axhline(0, c='k', lw=1, zorder=1)
			ax.set_axisbelow(True)
			ax.legend(fontsize='x-small', loc='upper right', ncol=2).remove()
		
		fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self._panel3)
		
		wpdc_oib_all_plot(wpdc_oib_df, ax1)
		wpdc_oib_day_plot(wpdc_oib_df, ax2)
		wpdc_oib_stock_plot(wpdc_oib_df, ax3)
		# ax.legend(ncol=2, fontsize='small', loc='lower right')
		# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
		fig.tight_layout()
		if save:
			plt.savefig(f"{figdir}\\Discovery_Scatter.pdf")
		fig.show() if show else plt.close()


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
		data['Volume Imbalance'] = (data['snap_bids'] - data['snap_asks']) * data['close_price'] / 10 ** 6
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
		data.loc[data.index.get_level_values(2) == 0, 'PDC'] = 0
		data['weights'] = abs(data['close_return'] / data['close_return_abs_total'])
		data['WPDC'] = data['weights'] * data['PDC']
		self._raw_data = data.join(data['WPDC'].groupby(['Date', 'Lag']).sum(), rsuffix='interval', how='inner')
	
	def plot_months_lags(self, save: bool = False, show: bool = True) -> None:
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
			sns.lineplot(data=tmp, x='Lag', y='Volume Imbalance', hue='Month', ci=None, estimator='sum',
			             palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Total market-wide volume order imbalance per day")
			ax.set_ylabel("Volume imbalance [million CHF]")
		
		def price_average_plot(ax):
			sns.lineplot(data=tmp, x='Lag', y='Deviation', hue='Month', ci=None,
			             palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Average price deviation per SLI title per day")
			ax.set_ylabel("Price deviation [bps]")
			ax.set_ylim([-80, 100])
		
		def price_median_plot(ax):
			sns.lineplot(data=tmp, x='Lag', y='Deviation', hue='Month', ci=None, estimator='median',
			             palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Median price deviation per SLI title per day")
			ax.set_ylabel("Price deviation [bps]")
			ax.set_ylim([-80, 100])
		
		def wpdc_plot(ax):
			fundata = tmp[tmp['close_return'] != 0].groupby(['Date', 'Lag']).first().reset_index()
			sns.lineplot(data=fundata, x='Lag', y='WPDCinterval', hue='Month', ci=None, estimator='mean',
			             palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Average WPDC per 30-second interval across SLI titles")
			ax.set_ylabel("Average $WPDC_{d,i}$")
			ax.set_ylim([-2, 3])
		
		def median_wpdc_plot(ax):
			fundata = tmp[tmp['close_return'] != 0].groupby(['Date', 'Lag']).first().reset_index()
			sns.lineplot(data=fundata, x='Lag', y='WPDCinterval', hue='Month', ci=None, estimator='median',
			             palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Median WPDC per 30-second interval across SLI titles")
			ax.set_ylabel("Median $WPDC_{d,i}$")
			ax.set_ylim([-2, 3])
		
		base_plot(volume_plot, 'VolumePlotMonthly')
		base_plot(rel_oib_plot, 'RelativeOIBMonthly')
		base_plot(abs_oib_plot, 'AbsoluteOIBMonthly')
		base_plot(price_average_plot, 'PriceDeviationMonthly')
		base_plot(price_median_plot, 'MedianPriceDeviationMonthly')
		base_plot(wpdc_plot, 'WPDCMonthly')
		base_plot(median_wpdc_plot, 'WPDCMedianMonthly')
	
	def plot_stocks_lags(self, nst=8, save: bool = False, show: bool = True) -> None:
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
		
		def volume_oib_plot(ax):
			sns.lineplot(data=data, x='Lag', y='Volume Imbalance', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
			             ax=ax, palette='cubehelix', hue_order=stocks, ci=None)
			ax.set_title("Average volume order imbalance throughout closing auction for {n} largest SLI titles".format(n=nst))
			ax.set_ylabel("Volume imbalance [million CHF]")
		
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
			ax.set_ylim([-60, 60])
		
		def deviation_median_plot(ax):
			sns.lineplot(data=data, x='Lag', y='Deviation', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
			             ax=ax, palette='cubehelix', hue_order=stocks, ci=None, estimator='median')
			ax.set_ylabel('Median deviation from closing price [bps]')
			ax.set_title("Median deviation from closing price throughout closing auction for {n} largest SLI titles".format(n=nst))
			ax.set_ylim([-60, 60])
		
		def wpdc_plot(ax):
			tmp_data = data[(data['close_return'] != 0) & (data['Symbol'] != 'CFR')]
			sns.lineplot(data=tmp_data, x='Lag', y='WPDC', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
			             ax=ax, palette='cubehelix', hue_order=stocks, ci=None, estimator='mean')
			ax.set_ylabel('Average $WPDC_{s,i} (=WPDC_{d,s,i})$')
			ax.set_title("Average WPDC throughout closing auction for {n} largest SLI titles".format(n=nst))
			ax.set_ylim([-0.01, 0.07])
		
		def median_wpdc_plot(ax):
			tmp_data = data[(data['close_return'] != 0) & (data['Symbol'] != 'CFR')]
			sns.lineplot(data=tmp_data, x='Lag', y='WPDC', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
			             ax=ax, palette='cubehelix', hue_order=stocks, ci=None, estimator='median')
			ax.set_ylabel('Median $WPDC_{d,s,i}$')
			ax.set_title("Median WPDC throughout closing auction for {n} largest SLI titles".format(n=nst))
			ax.set_ylim([-0.01, 0.07])
		
		base_plot(turnover_plot, 'Turnover')
		base_plot(volume_oib_plot, 'AbsoluteTurnover')
		base_plot(relative_oib_plot, 'RelativeTurnover')
		base_plot(deviation_plot, 'AverageDeviation')
		base_plot(deviation_median_plot, 'MedianDeviation')
	
	def plot_stocks_within(self, nstocks=10, save: bool = False, show: bool = True) -> None:
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
			ax.legend(handles=handles[1:], labels=labels[1:], fontsize='small', ncol=6, loc='upper right')
			ax.set_xlim([0, 600])
			ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
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
			ax.set_ylabel("Deviation from closing price [bps]")
			ax.set_title("{0}: Deviation from closing price throughout auction (each line represents one trading day)".format(stock))
			ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
		
		def relative_oib_plot(data, ax, stock):
			sns.lineplot(data=data, x='Lag', y='Relative Imbalance', ax=ax, units='Date',
			             estimator=None, lw=0.35, palette='rainbow', hue='Month')
			# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
			ax.set_title("{0}: Relative order imbalance throughout auction (each line represents one trading day)".format(stock))
			ax.set_ylabel("Relative Imbalance")
			ax.set_ylim([-0.8, 0.8])
		
		def volume_oib_plot(data, ax, stock):
			sns.lineplot(data=data, x='Lag', y='Volume Imbalance', ax=ax, hue='Month',
			             estimator=None, lw=0.35, palette='rainbow', units='Date')
			ax.set_title("{0}: Volume order imbalance throughout auction (each line represents one trading day)".format(stock))
			ax.set_ylabel("Volume Imbalance [million CHF]")
			ax.set_ylim([-200, 300])
		
		def wpdc_plot(data, ax, stock):
			funcdata = data[data['close_return'] != 0]
			sns.lineplot(data=funcdata, x='Lag', y='WPDC', ax=ax, hue='Month',
			             estimator=None, lw=0.5, palette='rainbow', units='Date')
			ax.set_title("{0}: WPDC throughout auction (each line represents one trading day)".format(stock))
			ax.set_ylabel("$WPDC_{d,s,i}$")
			ax.set_ylim([-1, 1])
		
		for s in stocks:
			tmp = data[data['Symbol'] == s].sort_values('Date', ascending=True)
			base_plot(tmp, deviation_plot, s, 'Deviation')
			base_plot(tmp, relative_oib_plot, s, 'RelativeImbalance')
			base_plot(tmp, volume_oib_plot, s, 'AbsoluteImbalance')
			base_plot(tmp, wpdc_plot, s, 'WPDC')
	
	def plots_report(self, nst=8, save: bool = False, show: bool = True) -> None:
		figdir = f"{os.getcwd()}\\06 Figures"
		lw = self._lw
		ms = self._ms
		stocks = self._avg_turnover.index[:nst].to_list()
		
		data = self._raw_data.reset_index(inplace=False)
		stockdata = data[data['Symbol'].isin(stocks)]
		
		monthdata = self._raw_data.groupby(['Lag', 'Date']).sum().reset_index()
		monthdata['Month'] = monthdata['Date'].apply(lambda d: d.strftime('%B'))
		
		def price_median_stocks(ax):
			sns.lineplot(data=stockdata, x='Lag', y='Deviation', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
			             ax=ax, palette='cubehelix', hue_order=stocks, ci=None, estimator='median')
			ax.set_ylabel('Price Deviation [bps]')
			ax.set_title("{Panel A}: Median price deviation by stock and interval")
			ax.legend(fontsize='small', ncol=round(nst / 2), loc='upper right')
		
		def price_median_months(ax):
			sns.lineplot(data=monthdata, x='Lag', y='Deviation', hue='Month', ci=None, estimator='median',
			             palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("{Panel B}: Median price deviation by month and interval")
			ax.set_ylabel("Price deviation [bps]")
			ax.legend(fontsize='small', ncol=6, loc='upper right')
		
		# fig, axes = plt.subplots(2, 1, figsize=self._panel2, sharey=True)
		# price_median_stocks(axes[0])
		# price_median_months(axes[1])
		# for ax in axes:
		# 	ax.grid(which='major', axis='y')
		# 	ax.axhline(0, c='k', lw=1, zorder=1)
		# 	ax.set_xlim((0, 600))
		# 	ax.set_xlabel("Seconds since auction start")
		# 	ax.set_ylim((-50, 50))
		# 	ax.set_axisbelow(True)
		# 	ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
		# 	ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
		#
		# fig.tight_layout()
		# if save:
		# 	plt.savefig(f"{figdir}\\Interval_Deviations.pdf")
		# plt.show() if show else plt.close()
		
		def wpdc_stocks(ax):
			funcdata = stockdata[(stockdata['close_return'] != 0) & (stockdata['Symbol'] != 'CFR')]
			sns.lineplot(data=funcdata, x='Lag', y='WPDC', hue='Symbol', lw=lw, marker='.', ms=ms, mew=0,
			             ax=ax, palette='cubehelix', hue_order=stocks, ci=None, estimator='mean')
			ax.set_ylabel('$WPDC^{IV}_{d,s,i}$')
			ax.set_title("Panel A: Average $WPDC^{IV}_{d,s,i}$ by stock and interval")
			ax.legend(fontsize='small', ncol=round(nst / 2), loc='upper right')
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
		
		def wpdc_months(ax):
			fundata = monthdata[monthdata['close_return'] != 0]
			sns.lineplot(data=fundata, x='Lag', y='WPDC', hue='Month', ci=None, estimator='median',
			             palette='cubehelix_r', lw=lw, ax=ax, marker='.', ms=ms, mew=0)
			ax.set_title("Panel B: Median $WPDC^{IV}_{d,i}$ by month and interval")
			ax.set_ylabel("$WPDC^{IV}_{d,i}$")
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
			ax.legend(fontsize='small', ncol=4, loc='upper right')
		
		fig, axes = plt.subplots(2, 1, figsize=self._panel2)
		wpdc_stocks(axes[0])
		wpdc_months(axes[1])
		for ax in axes:
			ax.grid(which='major', axis='y')
			ax.axhline(0, c='k', lw=1, zorder=1)
			ax.set_xlim((0, 600))
			ax.set_xlabel("Seconds since auction start")
			ax.set_axisbelow(True)
			ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
		
		fig.tight_layout()
		if save:
			plt.savefig(f"{figdir}\\Interval_WPDC.pdf")
		plt.show() if show else plt.close()


class ExtraVisual():
	def __init__(self, pricepath: str, volpath: str):
		self._figsize = (22 / 2.54, 13 / 2.54)
		self._stocklist = ['ABBN', 'CFR', 'LHN', 'NESN', 'NOVN', 'ROG', 'SREN', 'UBSG', 'ZURN']
		self._cwd = os.getcwd() + "\\03 Presentation April\\Extras"
		
		self._prices = pd.read_csv(pricepath, parse_dates=['onbook_date'])
		self._prices.rename(columns={'onbook_date': 'Date', 'price_org_ccy': 'Price'}, inplace=True)
		self._prices = self._prices[self._prices['symbol'].isin(self._stocklist)]
		
		self._vols = pd.read_csv(volpath, parse_dates=['onbook_date'])
		self._vols.rename(columns={'onbook_date': 'Date', 'bs_code': 'Side', 'old_volume': 'Volume'}, inplace=True)
		self._vols = self._vols.set_index(['Date', 'symbol', 'Side']).unstack(level='Side')
		self._vols.loc[:, ('Volume', 'S')] = self._vols.loc[:, ('Volume', 'S')] * (-1)
		self._vols.loc[:, ('Volume', 'Difference')] = self._vols.sum(axis=1)
		self._vols = self._vols.stack(level=1)
	
	def plot_extras(self, save: bool = False, show: bool = True) -> None:
		def pricefunc():
			fig, ax = plt.subplots(3, 3, figsize=self._figsize, dpi=450, sharex=True)
			for loc, stock in zip(itertools.product([0, 1, 2], [0, 1, 2]), self._stocklist):
				sns.lineplot(x='Date', y='Price', data=self._prices[self._prices['symbol'] == stock],
				             ci=None, ax=ax[loc[0], loc[1]])
				ax[loc[0], loc[1]].set_title(stock)
				ax[loc[0], loc[1]].set_xlabel("")
				ax[loc[0], loc[1]].set_ylabel("")
				ax[loc[0], loc[1]].set_ylim(bottom=0)
				ax[loc[0], loc[1]].grid(which='major', axis='both')
				ax[loc[0], loc[1]].axvline(x=pd.datetime(2019, 12, 31), lw=1.5, c='k', ls='dashed')
			ax[2, 2].set_xlim([pd.datetime(2019, 1, 1), pd.datetime(2020, 5, 1)])
			loca = dates.MonthLocator(bymonth=[1, 4, 7, 10, 13])
			form = dates.ConciseDateFormatter(loca, show_offset=False)
			ax[2, 2].xaxis.set_major_locator(loca)
			ax[2, 2].xaxis.set_major_formatter(form)
			fig.suptitle("\\textbf{Closing prices of major SLI titles in CHF}", size=12)
			fig.tight_layout(rect=[0, 0, 1, 0.96])
			if save:
				plt.savefig(self._cwd + "\\Prices.png", transparent=True)
			fig.show() if show else fig.close()
		
		def oibfunc(sharey: bool):
			fig, ax = plt.subplots(3, 3, figsize=self._figsize, dpi=450, sharex=True, sharey=sharey)
			for loc, stock in zip(itertools.product([0, 1, 2], [0, 1, 2]), self._stocklist):
				data = self._vols.xs(stock, level='symbol').reset_index()
				sns.lineplot(ax=ax[loc[0], loc[1]], x='Date', y='Volume', hue='Side', data=data)
				ax[loc[0], loc[1]].set_title(stock)
				ax[loc[0], loc[1]].set_xlabel("")
				ax[loc[0], loc[1]].set_ylabel("")
				ax[loc[0], loc[1]].grid(which='major', axis='both')
				ax[loc[0], loc[1]].legend().remove()
				ax[loc[0], loc[1]].axhline(y=0, lw=1, c='k')
				ax[loc[0], loc[1]].axvline(x=pd.datetime(2019, 12, 31), lw=1.5, c='k', ls='dashed')
			
			ax[2, 2].set_xlim([pd.datetime(2019, 1, 1), pd.datetime(2020, 5, 1)])
			fig.suptitle("\\textbf{Bid- [Red], Ask- [Blue] and Total [Green] Volume \"Overhang\" in CHF}", size=12)
			loca = dates.MonthLocator(bymonth=[1, 4, 7, 10, 13])
			form = dates.ConciseDateFormatter(loca, show_offset=False)
			ax[2, 2].xaxis.set_major_locator(loca)
			ax[2, 2].xaxis.set_major_formatter(form)
			fig.tight_layout(rect=[0, 0, 1, 0.96])
			if save:
				plt.savefig(self._cwd + "\\Volumes_{}.png".format(sharey), transparent=True)
			fig.show() if show else fig.close()
		
		pricefunc()
		oibfunc(True)
		oibfunc(False)
