import pandas as pd
from time import time
import os


class Research:
	def __init__(self, file_snapshots):
		t0 = time()
		self._snapbook = pd.read_csv(file_snapshots, header=0)
		self._symbols = self._snapbook['symbol'].unique()
		self._dates = self._snapbook['onbook_date'].unique()
		self._snapbook.set_index(['onbook_date', 'symbol', 'price'], drop=True, inplace=True)
		self._snapbook.sort_index(inplace=True)
		
		self._result_dict = {}  # Collects all the results
		
		print(">>> Class initiated ({} seconds)".format(round(time() - t0, 2)))
	
		
	@staticmethod
	def _extract_market_orders(imp_df):
		try:
			mark_buy = imp_df.loc[0, 'close_vol_bid']
		except KeyError:
			imp_df.loc[0, :] = 0
			mark_buy = imp_df.loc[0, 'close_vol_bid']
		
		try:
			mark_sell = imp_df.loc[0, 'close_vol_ask']
		except KeyError:
			imp_df.loc[0, :] = 0
			mark_sell = imp_df.loc[0, 'close_vol_ask']
		
		df = imp_df.drop(0, axis=0).sort_index()
		
		return df, mark_buy, mark_sell
	
	def results_to_df(self, indexnames):
		"""
		Export such that it can be used further in later stages.
		"""
		df = pd.DataFrame.from_dict(self._result_dict, orient='index')
		df.index.set_names(indexnames, inplace=True)
		return df
	
	def export_results(self, filename, filetype, indexes):
		df = self.results_to_df(indexes)
		if filetype == 'xlsx':
			df.round(4).to_excel(os.getcwd() + "\\Export\\{}.xlsx".format(filename))
		elif filetype == 'csv':
			df.round(4).to_csv(os.getcwd() + "\\Export\\{}.csv".format(filename))
	
	def get_SB(self):
		return self._snapbook
