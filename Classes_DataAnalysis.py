import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

conv = 2.54
dpi = 400
figsize = (22 / conv, 13 / conv)
figdir = os.getcwd() + "\\01 Presentation December\\Figures"


class DataAnalysis:
	def __init__(self, datapath, bluechippath):
		self._bluechips = pd.read_csv(bluechippath)['symbol']
		self._raw_data = pd.read_csv(datapath)
		self._figpath = os.getcwd() + "\\01 Presentation December"

	def _raw_to_bcs(self):
		self._bcs_data = self._raw_data[self._raw_data.index.get_level_values(level='Symbol').isin(self._bluechips)]


class SensAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._raw_data.set_index(['Mode', 'Date', 'Symbol', 'Percent'], inplace=True)
		self._raw_to_bcs()

	def plot_cont_sens(self):
		self._raw_data

	def plot_stock(self, stock):
		self._bcs_data.xs(stock, level='Symbol')


class DiscoAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._raw_data.set_index(['Date', 'Symbol'], inplace=True)
		self._raw_to_bcs()

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

		print(tmp_vol.tail(10))
# print(meanvol.sort_values(ascending=False))


file_data = os.getcwd() + "\\Exports\\Price_Discovery_v1.csv"
Discovery = DiscoAnalysis(file_data, file_bcs)
Discovery.plot_closings_per_stock()
