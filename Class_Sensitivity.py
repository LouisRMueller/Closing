import pandas as pd
import numpy as np
import copy
import time
from collections import deque
from Class_Research import Research


class SensitivityAnalysis(Research):
	@staticmethod
	def _calc_uncross(imp_df):
		"""
		Function calculates the theoretical uncross price of a closing order book.
		:return: dict()
		"""
		df, mark_buy, mark_sell = Research._extract_market_orders(imp_df)
		
		if 0 in df[['close_vol_ask', 'close_vol_bid']].sum().values:  # Where one side is empty
			return dict(price=np.nan, imbalance=np.nan, trade_vol=np.nan)
		
		else:
			n_lim = df.shape[0]
			limit_bids, limit_asks = deque(df['close_vol_bid'], n_lim), deque(df['close_vol_ask'], n_lim)
			
			neg_asks = limit_asks.copy()
			neg_asks.appendleft(0)
			
			cum_bids = mark_buy + np.cumsum(limit_bids)
			cum_asks = mark_sell + sum(limit_asks) - np.cumsum(neg_asks)
			
			total = cum_bids + cum_asks
			i = np.argmax(abs(total))
			
			output = dict(price=df.index[i], imbalance=total[i], trade_vol=min(cum_bids[i], cum_asks[i]))
			return output
	
	def _remove_liq(self, date, title, percentage=0, market=None, side=None):
		"""
		This function removes a certain percentage of liquidity from the closing auction.
		It is called for a every date-title combination individually
		:param date: Onbook_date
		:param title: Name of the stock
		:param percentage: Values in decimals, i.e. 5% is handed in as 0.05
		:param side: ['bid','ask','all']. Which side should be included in the removal
		:param market: True if market orders are included and False otherwise
		:return: A dateframe with new bid-ask book based on removing adjustments.
		"""
		imp_df = copy.deepcopy(self._snapbook.loc[(date, title), :])
		imp_df.replace({np.nan: 0}, inplace=True)
		
		if percentage == 0:
			return imp_df
		
		bids = imp_df['close_vol_bid'].tolist()
		asks = imp_df['close_vol_ask'].tolist()
		
		if market == "remove_all":  # Removes all market orders in closing auction
			ret_df = imp_df.loc[:, ('close_vol_ask', 'close_vol_bid')]
			try:
				ret_df.loc[0, :] = 0
				return ret_df
			except KeyError:
				return ret_df
		
		else:  # Only considering limit orders for adjustments
			removable_bid = sum(bids[1:]) * percentage
			removable_ask = sum(asks[1:]) * percentage
			imp_df.drop(columns=['cont_vol_bid', 'cont_vol_ask'], inplace=True)
		
		# Below is the algorithm
		if side in ['bid', 'all']:
			b = len(bids) - 1
			# remaining_liq = removable_bid / percentage
			while removable_bid > 0:
				local_vol = bids[b]
				bids[b] = local_vol - min(local_vol, removable_bid)
				removable_bid -= min(removable_bid, local_vol)
				b -= 1
		
		if side in ['ask', 'all']:
			a = 1
			# remaining_liq = removable_ask / percentage
			while removable_ask > 0:
				local_vol = asks[a]
				asks[a] = local_vol - min(local_vol, removable_ask)
				removable_ask -= min(removable_ask, local_vol)
				a += 1
		
		elif side == 'both':
			pass
		
		ret_df = pd.DataFrame([asks, bids], index=imp_df.columns, columns=imp_df.index).T
		return ret_df
	
	def sens_analysis(self, key, percents=tuple([1])):
		"""
		This function is supposed to exeucte the required calculations and add it to an appropriate data format.
		It calls other helper functions in order to determine the results of the analysis.
		:param key: String with mode of calculation ['bid_limit','ask_limit','all_limit','all_market','cont_market']
		:param percents: Iterable object with integers of percentages to be looped through.
		"""
		
		if key == 'bid_limit':
			side, mkt = 'bid', None
		elif key == 'ask_limit':
			side, mkt = 'ask', None
		elif key == 'all_limit':
			side, mkt = 'all', None
		elif key == 'all_market':
			side, mkt = 'all', 'remove_all'
		# elif key == "cont_market":
		# 	side, mkt = 'all', 'remove_cont'
		else:
			raise ValueError("key input not in ['bid_limit','ask_limit','all_limit','all_market','cont_market']")
		
		for date in self._dates:
			t0 = time.time()
			current_symbols = self.get_SB().loc[date, :].index.get_level_values(0).unique()
			
			for symbol in current_symbols:
				close_out = self._remove_liq(date=date, title=symbol, percentage=0)
				close_uncross = self._calc_uncross(close_out)
				
				for p in percents:
					res = self._result_dict[key, date, symbol, p] = {}
					
					remove_out = self._remove_liq(date=date, title=symbol, percentage=p, side=side, market=mkt)
					remove_uncross = self._calc_uncross(remove_out)
					
					res['close_price'] = close_uncross['price']
					res['close_vol'] = close_uncross['trade_vol']
					res['close_imbalance'] = close_uncross['imbalance']
					
					res['adj_price'] = remove_uncross['price']
					res['adj_vol'] = remove_uncross['trade_vol']
					res['adj_imbalance'] = remove_uncross['imbalance']
			
			print(">> {} finished ({} seconds)".format(date, round(time.time() - t0, 2)))
