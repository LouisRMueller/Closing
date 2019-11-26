import pandas as pd

class DataAnalysis:
	def __init__(self, datapath, bluechippath):
		self._bluechips =pd.read_csv(bluechippath, index_col='symbol')
		self._data = pd.read_csv(datapath)
		self._data = self._data[self._data['Symbol'].isin(self._bluechips.index)]
		
		
class SensAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._data.set_index(['Mode', 'Date', 'Symbol', 'Percent'], inplace=True)
		
class EffAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._data.set_index(['Date', 'Symbol'], inplace=True)
		