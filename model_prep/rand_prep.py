import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def Xy_prep_random_50(df, target, train_amount=50):
    scaler = StandardScaler()
    drop_cols = ['PID', 'SID', 'target_05', 'target_10', 'target_20', 'X', 'Y','Z', 'segment']
    df_1i = df.set_index('millisecond')
    cut_rows = int(target[-2:])*5
    pids = list(df['PID'].unique())
    train_par = list(np.random.choice(pids, train_amount, replace=False))
    test_par = pids
    for par in train_par:
        if par in test_par:
            test_par.remove(par)
    X_train = df_1i.loc[df_1i['PID'].isin(train_par), :].drop(columns=drop_cols).to_numpy()[:len(df)-cut_rows]
    y_train = df_1i.loc[df_1i['PID'].isin(train_par), target].to_numpy()[:len(df)-cut_rows]
    
    X_test = df_1i.loc[~df_1i['PID'].isin(train_par), :]
    test_par = list(X_test['PID'].unique())
    X_test = X_test.drop(columns=['SID', 'target_05', 'target_10', 'target_20', 'X', 'Y','Z', 'segment'])
    PID = X_test['PID']
    X_test.drop(columns=['PID'], inplace=True)
    X_train = scaler.fit_transform(X_train)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    X_test = X_test.merge(PID.reset_index(), left_index=True, right_index=True)
    X_test.set_index('millisecond', inplace=True)
    y_test = df_1i.loc[~df_1i['PID'].isin(train_par), [target, 'PID']]
    cols = df_1i.drop(columns=drop_cols).columns
    
    return X_train, X_test, y_train, y_test, train_par, test_par, cols, cut_rows