import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import dates
from matplotlib import ticker
from matplotlib.colors import LogNorm
import seaborn as sns
import copy

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# register_matplotlib_converters()

conv = 2.54
dpi = 300
figsize = (22 / conv, 13 / conv)
figdir = os.getcwd() + "\\01 Presentation December\\Figures"

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
def_palette = "cubehelix"
sns.set_palette(def_palette, desat=0.8)
def_color = sns.color_palette(def_palette, 1)[0]

