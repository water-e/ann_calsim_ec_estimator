#!/usr/bin/env python
# -*- coding: utf-8 -* 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def days_to_ops(ts):
    ts_out = ts*0 + 1.
    ts_out.values.fill(1.)
    ts_out=ts_out.where(ts.index.day <= ts, 0.)
    return ts_out



inputs="inputs_rs.csv"
outfile="inputs_rs2.csv"
smallfile="inputs_rs_flt.csv"
inputs="inputs_emm.csv"
outfile="inputs_emm2.csv"
smallfile="inputs_emm_flt.csv"
inputs="inputs_jp.csv"
outfile="inputs_jp2.csv"
smallfile="inputs_jp_flt.csv"



df = pd.read_csv(inputs,header=0,sep=",",index_col=0,parse_dates=[1])
df.to_csv(smallfile,float_format="%.2f",index=False)

df.set_index("date",drop=True,inplace=True)
orig_cols = df.columns
orig = df.dcc.copy()
df["dcc"] = days_to_ops(df.dcc)
for iday in range(1,8):
    df[f"dcc_{iday}d"] = df.dcc.shift(iday)

past = df.dcc.shift(7)
past = past.rolling(11).mean()
for iday in range(1,11):
    df[f"dcc_{iday}ave"] = past.shift((iday-1)*11+1)
df.to_csv("diagnose.csv",index=True,header=True,float_format="%.2f")
df=df[~df.index.isnull()]
print(df.index[df.index.duplicated()])
df.reset_index()
df=df.loc["1923-08-23":"2015-09-29",:]
df.to_csv(outfile,header=True,float_format="%.2f")
df["orig"] = np.nan
df["orig"].update(orig)
df[['orig','dcc','dcc_1d','dcc_2d','dcc_3d','dcc_4d','dcc_5d','dcc_6d','dcc_7d','dcc_1ave','dcc_2ave','dcc_10ave']].to_csv("try.csv",float_format="%.2f")




