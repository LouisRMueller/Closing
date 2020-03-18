@time using CSV, DataFrames, FixedEffectModels, RegressionTables, Dates

#%%
# IMPORT
Disco_df = CSV.read(pwd() * "\\Exports\\Price_Discovery_v3.csv")
Return_df = CSV.read(pwd() * "\\Data\\end_of_day_returns_bluechips.csv")
Volume_df = CSV.read(pwd() * "\\Data\\Open_Cont_Close_turnovers_2018-2019.csv")
Volume_df[!, :total_vol] = Volume_df[!, :opening_vol] .+ Volume_df[!, :continuous_vol] .+ Volume_df[!, :closing_vol]

#%%
# MANIPULATION
raw = join(Disco_df, Return_df, on = [:Date => :Date, :Symbol => :Symbol])
raw = join(raw, Volume_df, on = [:Date => :onbook_date, :Symbol => :symbol], kind = :left)
raw[raw[!, :pre_abs_spread].==0, :pre_abs_spread] .= 0.01
raw[!,:start_oib] = (raw[!,:start_bids] .- raw[!,:start_asks]) ./ (raw[!,:start_bids] .+ raw[!,:start_asks])
raw = join(raw, aggregate(raw[!, [:Date, :return_open_close]], :Date, x -> sum(abs.(x))), on = :Date)
raw[:, :PDC_stockday] = (raw[:, :return_close] ./ raw[:, :return_open_close])
raw[:, :WPDC_stockday] = abs.(raw[:, :return_open_close]) ./ raw[:, :return_open_close_function] .* raw[:, :PDC_stockday]
raw = join(raw, aggregate(raw[:, [:Date, :WPDC_stockday]], :Date, sum), on = :Date)
first(raw, 10)

#%%

# EXTRACTION
df = DataFrame(
    date = raw[!, :Date],
    dislocation1 = (raw[!, :actual_close_price] - raw[!, :pre_midquote]) ./ raw[!, :pre_midquote] * 100,
    dislocation2 = (raw[!, :actual_close_price] - raw[!, :pre_midquote]) ./ raw[!, :pre_abs_spread],
    symbol = categorical(raw[!, :Symbol]),
    start_oib = raw[!, :start_oib],
    turnover = raw[!, :close_vol],
    close_share = raw[!, :close_vol] ./ raw[!, :total_vol],
    return_20 = raw[!, :return_last_20],
    return_cont = raw[!, :return_cont],
    return_open_close = raw[!, :return_open_close],
    PDC_weight = raw[:, :return_close] ./ raw[:, :return_cont],
    PDC_stockday = raw[:, :PDC_stockday],
    WPDC_stockday = raw[:, :WPDC_stockday] .* 100,
)

df[!, :equivalence_dummy] = df[!, :date] .>= Date(2019, 7, 1)
df2 = df[df[:, :return_open_close].!=0, :]
df3 = df[df[:, :return_cont].!=0, :]

#%%

felm11 = reg(df, @formula(dislocation1 ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy))
felm12 = reg(df, @formula(dislocation1 ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy + fe(symbol)))
felm13 = reg(df, @formula(dislocation1 ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy + fe(symbol) + fe(date)))

felm21 = reg(df, @formula(dislocation2 ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy))
felm22 = reg(df, @formula(dislocation2 ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy + fe(symbol)))
felm23 = reg(df, @formula(dislocation2 ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy + fe(symbol) + fe(date)))

felm31 = reg(df3, @formula(PDC_weight ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy))
felm32 = reg(df3, @formula(PDC_weight ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy + fe(symbol)))
felm33 = reg(df3, @formula(PDC_weight ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy + fe(symbol) + fe(date)))

felm41 = reg(df2, @formula(WPDC_stockday ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy))
felm42 = reg(df2, @formula(WPDC_stockday ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy + fe(symbol)))
felm43 = reg(df2, @formula(WPDC_stockday ~ start_oib + close_share + return_20 + return_cont + equivalence_dummy + fe(symbol) + fe(date)))

#%%

regtable(felm11, felm12, felm13, felm21, felm22, felm23, felm31, felm32, felm33, felm41, felm42, felm43)
