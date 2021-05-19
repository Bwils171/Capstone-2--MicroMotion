import numpy as np
import pandas as pd

def Xy_prep_all(df, target, a=1, b=75):
    df_1 = df.loc[df['PID'].between(a, b)]
    drop_cols = ['PID', 'SID', 'target_05', 'target_10', 'target_20', 'X', 'Y','Z', 'segment']
    df_1i = df_1.set_index('millisecond')
    participants = b-a+1
    cut_rows = int(target[-2:])*participants*10
    X = df_1i.drop(columns=drop_cols).to_numpy()[:len(df)-cut_rows]
    y = df_1i[target].to_numpy()[:len(df)-cut_rows]
    cols = df_1i.drop(columns=drop_cols).columns
    participants = b-a+1
    end_train = int((len(X)) * (2/3))
    end_test = int(len(X))  
    X_train = X[:end_train, :]
    X_test = X[end_train:end_test, :]
    y_train = y[:end_train]
    y_test = y[end_train:end_test]
    
    return X_train, X_test, y_train, y_test, cols, participants, cut_rows

