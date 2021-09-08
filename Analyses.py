from Connection import Connection
from funcs_data import *
from funcs_stats import *


class GranularAnalysis(Connection):
    def __init__(self, import_source: str):
        if import_source == 'database':
            super().__init__()
        elif import_source == 'raw_local':
            self.master_data = self.import_master_local()
        else:
            raise Exception("Wrong input given")

        pd.set_option("display.max_columns", 17)
        pd.set_option('display.width', 300)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    def create_limit_plots(self, upper_limit: float = 0.5):
        control = {'execution': ['execution-bid-dev', 'execution-ask-dev'],
                   'liquidity': ['liquidity-bid-dev', 'liquidity-ask-dev']}
        titles = ["Panel A: Removal of bid liquidity", "Panel B: Removal of ask liquidity"]
        kwargs = {'showfliers': False, 'palette': 'cubehelix_r', 'hue': 'size-quantile', 'whis': [5, 95], 'linewidth': 1}
        data = prepare_data_for_latex_plot(self.master_data.loc[pd.IndexSlice[:, :, 0.05:upper_limit + 0.01], :])

        for c in control:
            fig, ax = plt.subplots(2, 1, figsize=self._figsizeDouble)
            for i in np.arange(len(control[c])):
                sns.boxplot(data=data, x='removal', y=control[c][i], ax=ax[i], **kwargs)
                ax[i].xaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: f"{int((pos + 1) * 5)}\%"))
                draw_gridlines(ax[i], axis='y')
                ax[i].set_ylabel("Absolute deviation [bps]")
                ax[i].axhline(y=0, **self._helpLineKwargs)
                ax[i].legend(title='Size quantile')
                ax[i].set_xlabel("Liquidity removal [\%]")
                ax[i].set_title(titles[i])
                # ax[i].set_ylim(bottom=0, top=200)
            fig.tight_layout()
            show_and_save_graph(f"{c}_removal")

    def create_market_plots(self, upper_limit: float = 0.5):
        c = 'market'
        control = {'market': ['market-bid-dev', 'market-ask-dev', 'market-both-dev']}

        titles = ["Panel A: Removal of bid market orders", "Panel B: Removal of ask market orders", "Panel C: Removal of bid and ask market orders"]
        kwargs = {'showfliers': False, 'palette': 'cubehelix_r', 'hue': 'size-quantile', 'whis': [5, 95], 'linewidth': 1}
        data = prepare_data_for_latex_plot(self.master_data.loc[pd.IndexSlice[:, :, 0.05:upper_limit + 0.01], :])

        fig, ax = plt.subplots(2, 1, figsize=self._figsizeDouble)
        for i in np.arange(2):
            sns.boxplot(data=data, x='removal', y=control[c][i], ax=ax[i], **kwargs)
            ax[i].xaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: f"{int((pos + 1) * 5)}\%"))
            draw_gridlines(ax[i], axis='y')
            ax[i].set_ylabel("Absolute deviation [bps]")
            ax[i].axhline(y=0, **self._helpLineKwargs)
            ax[i].legend(title='Size quantile')
            ax[i].set_xlabel("Liquidity removal [\%]")
            ax[i].set_title(titles[i])
            # ax[i].set_ylim(bottom=0)
        fig.tight_layout()
        show_and_save_graph(f"{c}_removal")

        # data2 = self.master_data.xs(1, level=-1).set_index('size_quantile', append=True)[['market_bid_dev', 'market_ask_dev', 'market_both_dev']]
        # data2 = pd.melt(np.abs(data2), ignore_index=False, var_name='mode')
        # data2 = prepare_data_for_latex_plo§t(data2)
        # fig, ax = plt.subplots(1, 1, figsize=self._figsizeNorm)
        # sns.boxplot(data=data2, x='mode', y='value', ax=ax, **kwargs)
        # ax.axhline(y=0, **self._helpLineKwargs)
        # ax.set_ylim(bottom=0)
        # ax.legend(title='Size quantile')
        # ax.set_xticklabels(['Bids only', 'Asks only', 'Bids and asks'])
        # ax.set_xlabel("")
        # ax.set_ylabel("Absolute deviation [bps]")
        # draw_gridlines(ax)
        # fig.tight_layout()
        # show_and_save_graph(f"all_market_removal")

        data3 = self.master_data.xs(1, level='removal')[['market_bid_dev', 'market_ask_dev', 'market_both_dev', 'size_quantile']]
        data3['x'] = 1
        data3 = prepare_data_for_latex_plot(np.abs(data3))
        names = ['Bid removal', 'Ask removal', 'Bid + Ask removal']
        fig, ax = plt.subplots(1, 3, figsize=self._figsizeNorm)
        for i in np.arange(len(control[c])):
            sns.boxplot(data=data3, y=control[c][i], x='x', **kwargs, ax=ax[i])
            ax[i].legend().remove()
            ax[i].xaxis.set_major_locator(tick.NullLocator())
            ax[i].yaxis.set_major_locator(tick.MaxNLocator(10))
            ax[i].set_xlabel("")
            ax[i].set_ylim(bottom=0)
            ax[i].set_title(names[i])
            ax[i].set_ylabel("Absolute Deviation [bps]")
            draw_gridlines(ax[i], 'y')
            if i == 2:
                ax[i].set_ylim([0,100])
            else:
                ax[i].set_ylim([0,200])

        plt.tight_layout()
        show_and_save_graph(f"all_market_removal_facet")


class OvernightAnalysis(Connection):
    def __init__(self, import_source: str):
        pd.set_option("display.max_columns", 17)
        pd.set_option('display.width', 300)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        self._mlist = list()

        if import_source in 'database':
            super().__init__()
            self.data = self.manipulate_data(self.master_data)
            del self.master_data
        elif import_source == 'raw_local':
            master_data = self.import_master_local()
            self.data = self.manipulate_data(master_data)
        elif import_source == 'refined_local':
            self.data = self.import_refined_data()
        else:
            raise Exception("Wrong input given")

        # print_stockwise_stats(self.data)

    def analyse_overnight_volatility(self):
        def estimate_panel(frame: pd.DataFrame, dep: str, con: list[str] = (), ind: list[str] = (), qnt: list[str] = ()) -> None:
            variable_names: list[str] = [dep] + functools.reduce(lambda x, y: list(x) + list(y), [con, ind, qnt])
            check_column_names(frame, variable_names)

            if qnt:
                df_quantiled = interact_with_quantiled_series(frame, factor_names=qnt, quantiled_name='size_quantile')
                data = frame.join(df_quantiled, how='left')
                variable_names = list(filter(lambda x: x not in qnt, variable_names)) + df_quantiled.columns.to_list()
            else:
                data = frame

            self._mlist.extend(estimate_FamaMacBeth_panel(data.loc[:, variable_names]))
            self._mlist.extend(estimate_time_FE_panel(data.loc[:, variable_names]))
            # self._mlist.extend(estimate_stock_FE_panel(data.loc[:, variable_names]))

        df = self.data.copy()
        ret_names = list(filter(lambda x: '_to_' in x, df.columns))
        df[ret_names] = np.abs(df[ret_names])

        dep_names = list(filter(lambda x: '+' in x and x.startswith('close_to'), df.columns))
        con = ['open_to_close', 'preclose_to_close', 'monthly', 'quarterly']
        con = ['open_to_close', 'preclose_to_close']
        ind = ['abs_market_imbalance', 'market_buy_ratio', 'market_sell_ratio']

        for dep in dep_names:
            estimate_panel(df, dep=dep, ind=ind[:1], con=con)
        self.print_panel_output()

        for dep in dep_names:
            estimate_panel(df, dep=dep, qnt=ind[:1], con=con)
        self.print_panel_output()

        for dep in dep_names:
            estimate_panel(df, dep=dep, con=con, ind=ind[1:])
        self.print_panel_output()
        for dep in dep_names:
            estimate_panel(df, dep=dep, con=con, qnt=ind[1:])
        self.print_panel_output()
        breakpoint()


    def print_panel_output(self) -> None:
        mlist = self._mlist
        if len(mlist) > 0:
            out = {f'({x + 1})': mlist[x] for x in range(len(mlist))}
            comp = lm.panel.compare(out, stars=True).summary
            # print(comp.as_latex())
            print(comp)

            mlist.clear()

    def plot_KDEs_market_ratios(self):
        cmap = 'Reds'
        df = self.data

        # fig, ax = plt.subplots(1, 1, figsize=self._figsizeNorm)
        # sns.kdeplot(data=prepare_data_for_latex_plot(df), x='market-buy-ratio', y='market-sell-ratio', fill=True, thresh=0.05, cmap=cmap)
        # fig.show()

        g = sns.FacetGrid(data=prepare_data_for_latex_plot(df), col='size-quantile', sharey='row', sharex=True, margin_titles=False, despine=False,
                          xlim=(0, 1), ylim=(0, 1), hue='size-quantile', col_wrap=2, height=3, aspect=1)
        g.map(plt.grid, axis='both', which='major', lw=0.6, color='black')
        g.map_dataframe(sns.kdeplot, x='market-buy-ratio', y='market-sell-ratio', fill=True, levels=np.linspace(0.05, 1.0, 10), cmap=cmap)
        show_and_save_graph("Market_Ratios_KDE")

    def plot_WPDC_normalized(self):
        legend_loc, xlabel, ncol = 'lower left', 'Market order imbalance', 2

        def consolidate(frame: pd.DataFrame, wpdc_name: str, consolidation: str) -> pd.DataFrame:
            grp = frame.groupby(consolidation)
            normalizer = lambda values: np.mean(values) / (np.std(values) / np.sqrt(len(values)))
            if consolidation == 'date':
                agg_dict = {wpdc_name: normalizer, 'market_imbalance': normalizer,
                            'closing_volume': np.sum}
            elif consolidation == 'symbol':
                agg_dict = {wpdc_name: normalizer, 'market_imbalance': normalizer,
                            'closing_volume': lambda values: np.log2(np.mean(values * 10 ** 6))}
            else:
                raise Exception(f"Wrong input given for {consolidation=}")
            new = grp.agg(agg_dict).reset_index()
            new.sort_values('closing_volume', ascending=True, inplace=True)
            return new

        df1 = self.data[self.data.closing_volume > 1].reset_index()

        df1.sort_values('closing_volume', ascending=True, inplace=True)
        df2 = consolidate(df1, 'WPDC_backward', 'date')
        df3 = consolidate(df1, 'WPDC_backward', 'symbol')

        fig, (ax2, ax3) = plt.subplots(2, 1, figsize=self._figsizeDouble)

        sns.scatterplot(data=prepare_data_for_latex_plot(df2), x='market-imbalance', y='WPDC-backward', ax=ax2, size='closing-volume', ec='k',
                        sizes=(5, 400), legend='brief', hue='closing-volume', palette='cubehelix_r', zorder=2)
        ax2.set_ylim((-10, 10))
        ax2.set_xlim((-8, 8))
        draw_gridlines(ax2, 'both')
        ax2.set_title(f"Panel A: $WPDC^{{CL}}_{{d}}$ and average order imbalance by day ($n = {len(df2)}$)")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("$WPDC^{CL}_{d}$")
        ax2.legend(loc=legend_loc, title='Total turnover [mn.]', ncol=ncol, fontsize='small')
        ax2.yaxis.set_major_locator(tick.MultipleLocator(2))

        sns.scatterplot(data=prepare_data_for_latex_plot(df3), x='market-imbalance', y='WPDC-backward', ax=ax3, size='closing-volume',
                        hue='closing-volume', ec='k', palette='cubehelix_r', sizes=(10, 400), zorder=2, legend='brief')
        ax3.set_title(f"Panel B: $TPDC^{{CL}}_{{s}}$ and average order imbalance by stock ($n = {len(df3)}$)")
        ax3.set_xlim((-8, 8))
        ax3.set_ylim((-10, 10))
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel("$TPDC^{CL}_{s}$")
        draw_gridlines(ax3, 'both')
        ax3.legend(loc=legend_loc, title='$\log_2 (avg. turnover)$', ncol=ncol, fontsize='small')
        ax3.yaxis.set_major_locator(tick.MultipleLocator(2))

        for ax in (ax2, ax3):
            borders, l = (-2, 2), 0
            ax.axhline(y=l, **self._helpLineKwargs)
            ax.axvline(x=l, **self._helpLineKwargs)
            ax.axhspan(borders[0], borders[1], **self._shadingKwargs)
            ax.axvspan(borders[0], borders[1], **self._shadingKwargs)

        fig.tight_layout()
        show_and_save_graph("Scatterplots_Normalized")


    def DEPR_analyse_overnight_returns(self):
        """Overnight returns are unpredictable, however, they reverse the closing return by around 30%"""
        def estimate_panel(frame: pd.DataFrame, dep: str, con: list[str] = (), ind: list[str] = (), qnt: list[str] = ()) -> None:
            variable_names: list[str] = [dep] + functools.reduce(lambda x, y: list(x) + list(y), [con, ind, qnt])
            check_column_names(frame, variable_names)

            if qnt:
                df_quantiled = interact_with_quantiled_series(frame, factor_names=qnt, quantiled_name='size_quantile')
                data = frame.join(df_quantiled, how='left')
                variable_names = list(filter(lambda x: x not in qnt, variable_names)) + df_quantiled.columns.to_list()
            else:
                data = frame

            self._mlist.extend(estimate_time_FE_panel(data.loc[:, variable_names]))
            self._mlist.extend(estimate_stock_FE_panel(data.loc[:, variable_names]))
            self._mlist.extend(estimate_FamaMacBeth_panel(data.loc[:, variable_names]))

        df = self.data.copy()
        # ret_names = list(filter(lambda x: '_to_' in x, df.columns))
        # df[ret_names] = np.abs(df[ret_names])

        dep_names = list(filter(lambda x: '+' in x and x.startswith('close_to'), df.columns))
        con = ['open_to_close', 'preclose_to_close', 'monthly', 'quarterly']
        con = ['open_to_close', 'preclose_to_close']
        ind = ['market_imbalance', 'preclose_imbalance', 'market_buy_ratio', 'market_sell_ratio']

        for dep in dep_names:
            estimate_panel(df, dep=dep, ind=ind[:1], con=con)
            estimate_panel(df, dep=dep, qnt=ind[:1], con=con)
        self.print_panel_output()

        for dep in dep_names:
            estimate_panel(df, dep=dep, con=con, ind=ind[1:])
            estimate_panel(df, dep=dep, con=con, qnt=ind[1:])
        self.print_panel_output()
        breakpoint()



class DeprecatedMethods(OvernightAnalysis, GranularAnalysis):
    def DEPR_analyse_WPDC_backward(self):
        def estimate_panel(frame: pd.DataFrame, dep: str, con: list[str] = (), ind: list[str] = (), qnt: list[str] = ()) -> None:
            variable_names: list[str] = [dep] + functools.reduce(lambda x, y: list(x) + list(y), [con, ind, qnt])
            check_column_names(frame, variable_names)

            if qnt:
                df_quantiled = interact_with_quantiled_series(frame, factor_names=qnt, quantiled_name='size_quantile')
                data = frame.join(df_quantiled, how='left')
                variable_names = list(filter(lambda x: x not in qnt, variable_names)) + df_quantiled.columns.to_list()
            else:
                data = frame

            self._mlist.extend(estimate_time_FE_panel(data.loc[:, variable_names]))
            self._mlist.extend(estimate_stock_FE_panel(data.loc[:, variable_names]))

        df = self.data.copy()
        # con = ['open_to_close', 'preclose_to_close', 'monthly', 'quarterly']
        con = ['open_to_close', 'preclose_to_close', 'continuous_volume', 'closing_volume', 'quarterly', 'monthly']
        ind = ['market_imbalance', 'abs_market_imbalance', 'market_buy_ratio', 'market_sell_ratio']
        dep = 'WPDC_backward'

        estimate_panel(df, dep=dep, con=con)
        estimate_panel(df, dep=dep, con=con, ind=ind[:1])
        estimate_panel(df, dep=dep, con=con, ind=ind[1:2])
        estimate_panel(df, dep=dep, con=con, ind=ind[-2:])
        self.print_panel_output()

        breakpoint()

    def DEPR_plot_WPDC_raw(self):
        def consolidate(frame: pd.DataFrame, wpdc_name: str, consolidation: str, ) -> pd.DataFrame:
            grp = frame.groupby(consolidation)
            if consolidation == 'date':
                # agg_dict = {wpdc_name: np.sum, 'market_imbalance': np.mean,
                #             'closing_volume': lambda values: np.mean(np.log(values * 1_000_000))}
                agg_dict = {wpdc_name: np.sum, 'market_imbalance': np.mean,
                            'closing_volume': np.sum}
            elif consolidation == 'symbol':
                agg_dict = {wpdc_name: lambda values: np.mean(values) / (np.std(values) / np.sqrt(len(values))),
                            'market_imbalance': np.mean,
                            'closing_volume': np.mean}
                agg_dict = {wpdc_name: lambda values: np.mean(values) / (np.std(values) / np.sqrt(len(values))),
                            'market_imbalance': np.mean,
                            'closing_volume': lambda values: np.log(np.mean(values * 10 ** 6))}
            else:
                raise Exception(f"Wrong input given for {consolidation=}")
            new = grp.agg(agg_dict).reset_index()
            new.sort_values('closing_volume', ascending=True, inplace=True)
            return new

        df1 = self.data[self.data.closing_volume > 1].reset_index()

        df1['log_closing_volume'] = np.log(df1['closing_volume'] * 1_000_000)
        df1.sort_values('log_closing_volume', ascending=True, inplace=True)
        df2 = consolidate(df1, 'WPDC_backward', 'date')
        df3 = consolidate(df1, 'WPDC_backward', 'symbol')

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self._figsizeTriple)
        sns.scatterplot(data=df1, x='market_imbalance', y='WPDC_backward', ax=ax1, size='closing_volume',
                        hue='closing_volume', ec='k', palette='cubehelix_r', sizes=(10, 400), zorder=2)
        ax1.set_xlim((-1, 1))
        # ax1.set_title(r"Panel A: $WPDC^{{CL}}_{{d,s}}$ and order imbalance ($N = D \times S = 7470$)")
        # ax1.set_xlabel(xlabel)
        # ax1.set_ylabel("$WPDC^{CL}_{d,s}$")
        ax1.set_ylim((-0.03, 0.03))
        draw_gridlines(ax1, 'both')
        ax1.axvline(0, c='k', lw=1, zorder=1)
        ax1.axhline(0, c='k', lw=1, zorder=1)
        ax1.yaxis.set_major_locator(tick.MultipleLocator(0.01))

        sns.scatterplot(data=df2, x='market_imbalance', y='WPDC_backward', ax=ax2, size='closing_volume', ec='k',
                        sizes=(5, 500), legend='brief', hue='closing_volume', palette='cubehelix_r', zorder=2)
        ax2.set_xlim((-0.3, 0.3))

        ax2.set_ylim(bottom=-0.3, top=0.3)
        ax2.set_xlim((-0.25, 0.25))
        draw_gridlines(ax2, 'both')
        ax2.axvline(0, c='k', lw=1, zorder=1)
        ax2.axhline(0, c='k', lw=1, zorder=1)
        # ax2.grid(which='major', axis='both')
        # ax2.set_title(f"Panel B: $WPDC^{{CL}}_{{d}}$ and average order imbalance by day ($D = {tmp.shape[0]}$)")
        # ax2.set_xlabel(xlabel)
        # ax2.set_ylabel("$WPDC^{CL}_{d}$")
        # ax2.set_axisbelow(True)
        # ax2.legend(loc='lower right', title='Turnover [mn. CHF]', ncol=2, fontsize='small')
        # ax2.yaxis.set_major_formatter(tick.PercentFormatter(xmax=1, decimals=1))

        sns.scatterplot(data=df3, x='market_imbalance', y='WPDC_backward', ax=ax3, size='closing_volume', hue='closing_volume', ec='k',
                        palette='cubehelix_r', sizes=(10, 500), zorder=2, legend='brief')
        # ax3.set_title(f"Panel C: $TPDC^{{CL}}_{{s}}$ and average order imbalance by stock ($S = {tmp.shape[0]}$)")
        ax3.set_xlim((-0.05, 0.05))
        ax3.set_ylim((-8, 8))
        ax3.axhspan(-1.96, 1.96, alpha=0.2, color='grey')
        # ax3.set_xlabel(xlabel)
        ax3.set_ylabel("$TPDC^{CL}_{s}$")
        draw_gridlines(ax3, 'both')
        ax3.axvline(0, c='k', lw=1, zorder=1)
        ax3.axhline(0, c='k', lw=1, zorder=1)

        fig.tight_layout()
        fig.show()

    def DEPR_plot_KDEs_overnight_returns(self):
        df = self.data

        return_cols = list(filter(lambda x: 'close_to_' in x, df.columns))
        tmp = df.melt(id_vars=['size_quantile', 'market_imbalance'], value_vars=return_cols, var_name='mode',
                      ignore_index=False)
        # tmp = prepare_data_for_latex_plot()

        g = sns.FacetGrid(data=tmp, row='mode', sharey='row', sharex=True, margin_titles=True, despine=False, xlim=(-1, 1), ylim=(-10, 10))
        g.map(plt.axvline, x=0, **self._helpLineKwargs)
        g.map(plt.axhline, y=0, **self._helpLineKwargs)
        g.map_dataframe(sns.kdeplot, x='market_imbalance', y='value', fill=True, thresh=0.05, cmap='Reds')
        plt.gca().xaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: f"{int(x)}"))
        plt.show()

    def DEPR_plot_KDEs_market_imbalance(self):
        cmap = 'Blues'
        df = self.data
        x, y = 'market_imbalance', 'WPDC_backward'
        breakpoint()
        fig, ax = plt.subplots(1, 1, figsize=self._figsizeNorm)
        sns.kdeplot(data=df, x=x, y=y, fill=True, thresh=0.05, cmap=cmap, ax=ax)
        ax.set_ylim((-0.01, 0.01))
        ax.set_xlim((-1, 1))
        fig.show()

        g = sns.FacetGrid(data=df, col='size_quantile', sharey='row', sharex=True, margin_titles=False, despine=False,
                          xlim=(-1, 1), ylim=(-0.01, 0.01), hue='size_quantile', col_wrap=2, height=3, aspect=1)
        g.map(plt.grid, axis='both', which='major', lw=0.6, color='black')
        g.map_dataframe(sns.kdeplot, x=x, y=y, fill=True, levels=np.linspace(0.05, 1.0, 10), cmap=cmap)
        plt.show()
