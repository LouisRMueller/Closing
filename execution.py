from Analyses import *

# G = GranularAnalysis(import_source='database')
# G = GranularAnalysis(import_source='raw_local')
# G.create_limit_plots(upper_limit=0.35)
# G.create_market_plots(upper_limit=0.35)


# O = OvernightAnalysis(import_source='database')
O = OvernightAnalysis(import_source='raw_local')
# O = OvernightAnalysis(import_source='refined_local')
# O.analyse_overnight_volatility()
O.DEPR_analyse_overnight_returns()
# O.plot_WPDC_normalized()
O.plot_KDEs_market_ratios()
