import pandas as pd

from funcs_base import *
import linearmodels as lm
import statsmodels.api as sm


def estimate_FamaMacBeth_panel(data: pd.DataFrame) -> list:
    data = data.dropna()
    models = list()
    # models.append(lm.PanelOLS(data.iloc[:, 0], data.iloc[:, 1:], entity_effects=True).fit(cov_type='kernel'))
    # models.append(lm.PanelOLS(data.iloc[:, 0], data.iloc[:, 1:], time_effects=True).fit(cov_type='kernel'))
    # models.append(lm.PooledOLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:])).fit(cov_type='kernel'))
    models.append(lm.FamaMacBeth(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:])).fit(cov_type='kernel'))
    return models


def estimate_stock_FE_panel(data: pd.DataFrame) -> list:
    data = data.dropna()
    models = list()
    models.append(lm.PanelOLS(data.iloc[:, 0], data.iloc[:, 1:], entity_effects=True).fit(cov_type='kernel'))
    return models

def estimate_time_FE_panel(data: pd.DataFrame) -> list:
    data = data.dropna()
    models = list()
    models.append(lm.PanelOLS(data.iloc[:, 0], data.iloc[:, 1:], time_effects=True, drop_absorbed=True).fit(cov_type='kernel'))
    return models


def estimate_Between_panel(data: pd.DataFrame, cluster: str = 'date') -> list:
    grouped = data.groupby(cluster).mean()
    grouped.index = pd.MultiIndex.from_product([grouped.index.to_list(),[1]])
    models = list()
    models.append(lm.BetweenOLS(grouped.iloc[:, 0], sm.add_constant(grouped.iloc[:, 1:])).fit(cov_type='clustered'))
    return models


def interact_with_quantiled_series(frame: pd.DataFrame, factor_names: list[str], quantiled_name: str, letter: str = 'Q') -> pd.DataFrame:
    collector = list()
    quantiled_wide = pd.get_dummies(frame[quantiled_name])
    for factor_name in factor_names:  # Loop through factors
        new_names = list(map(lambda q: f"Q{q} x {factor_name}", quantiled_wide.columns))
        result = pd.DataFrame(np.multiply(frame[[factor_name]].to_numpy(), quantiled_wide.to_numpy()), columns=new_names, index=frame.index)
        collector.append(result)

    return pd.concat(collector, axis=1)


def add_interacted_column(frame: pd.DataFrame, first_name: str, second_name: str) -> None:
    frame[f"{first_name} x {second_name}"] = frame[first_name] * frame[second_name]
