@time using CSV, DataFrames, GLM, FixedEffectModels, RegressionTables, Dates

#%%

path = pwd() * "\\Exports\\Price_Discovery_v3.csv"
PD_raw = CSV.read(path)

df = DataFrame(
    date = PD_raw[!, :Date],
    dislocation1 = (PD_raw[!, :actual_close_price] - PD_raw[!, :pre_midquote]) ./ PD_raw[!, :pre_midquote],
    dislocation2 = (PD_raw[!, :actual_close_price] - PD_raw[!, :pre_midquote]) ./ PD_raw[!, :pre_abs_spread],
    symbol = categorical(PD_raw[!, :Symbol]),
    imbalance = missing,
    turnover = missing,
    return15 = missing,
)

df[!, :equivalence_dummy] = df[!, :date] .>= Date(2019, 7, 1)

#%%

lm1 = lm(@formula(pre_rel_spread ~ close_vol), raw)
felm1 = reg(raw, @formula(close_vol ~ pre_rel_spread + close_imbalance + fe(Symbol)), save = true)

regtable(lm1, felm1)

PD_raw[(PD_raw[:, :pre_abs_spread] .== 0) .& (PD_raw[:, :Date] .== Date(2019,12,20)) ,:]
