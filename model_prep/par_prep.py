import numpy as np
import pandas as pd

def Xy_prep_participants(df, target, a=50):
    drop_cols = ['PID', 'SID', 'target_05', 'target_10', 'target_20', 'X', 'Y','Z', 'segment']
    df_1i = df.set_index('millisecond')
    cut_rows = int(target[-2:])*20
    X_train = df_1i.loc[df_1i['PID'] <= a,:].drop(columns=drop_cols).to_numpy()[:len(df)-cut_rows]
    y_train = df_1i.loc[df_1i['PID'] <= a, target].to_numpy()[:len(df)-cut_rows]
    X_test = df_1i.loc[df_1i['PID'] > a,:].drop(columns=drop_cols).to_numpy()[:len(df)-cut_rows]
    y_test = df_1i.loc[df_1i['PID'] > a, target].to_numpy()[:len(df)-cut_rows]
    cols = df_1i.drop(columns=drop_cols).columns
    
    return X_train, X_test, y_train, y_test, cols, cut_rows