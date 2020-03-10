@time using CSV, DataFrames, GLM, FixedEffectModels, RegressionTables, Dates, DataTables

#%%

path1 = pwd() * "\\Exports\\Price_Discovery_v3.csv"
Disco_df = CSV.read(path1)

path2 = pwd() * "\\Data\\end_of_day_returns_bluechips.csv"
Return_df = CSV.read(path2)

raw = join(Disco_df, Return_df, on=[:Date => :onbook_date, :Symbol => :symbol])

df = DataFrame(
    date = raw[!, :Date],
    dislocation1 = (raw[!, :actual_close_price] - raw[!, :pre_midquote]) ./ raw[!, :pre_midquote],
    dislocation2 = (raw[!, :actual_close_price] - raw[!, :pre_midquote]) ./ raw[!, :pre_abs_spread],
    symbol = categorical(raw[!, :Symbol]),
    imbalance_pre = raw[:, :start_oib],
    turnover = raw[:, :close_vol]
)

df[!, :equivalence_dummy] = df[!, :date] .>= Date(2019, 7, 1)

# Join both dataframes together
df = join(df, Return_df, on = [:date => :onbook_date, :symbol => :symbol])


#%%

lm1 = lm(@formula(pre_rel_spread ~ close_vol), raw)
felm1 = reg(raw, @formula(close_vol ~ pre_rel_spread + close_imbalance + fe(Symbol)), save = true)

regtable(lm1, felm1)
