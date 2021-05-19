import numpy as np
import pandas as pd


def Xy_prep_music(df, target, a=1, b=75):
    df_1 = df.loc[df['PID'].between(a, b)]
    drop_cols = ['PID', 'SID', 'target_05', 'target_10', 'target_20', 'X', 'Y','Z', 'segment']
    df_1i = df_1.set_index('millisecond')
    participants = b-a+1
    cut_rows = int(target[-2:])*participants*10
    X = df_1i.drop(columns=drop_cols).to_numpy()[:len(df)-cut_rows]
    y = df_1i[target].to_numpy()[:len(df)-cut_rows]
    cols = df_1i.drop(columns=drop_cols).columns
    participants = b-a+1
    end_sil_test = int( len(X) / 2)
    start_music_test = int(end_sil_test + ((end_sil_test-cut_rows) * (2/3)))
    X_train = X[end_sil_test:start_music_test, :]
    X_test = X[start_music_test:len(X)-cut_rows, :]
    y_train = y[end_sil_test:start_music_test]
    y_test = y[start_music_test:len(X)-cut_rows]    
    
    return X_train, X_test, y_train, y_test, cols, participants, cut_rows
