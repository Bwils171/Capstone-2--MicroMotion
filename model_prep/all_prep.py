
def Xy_prep_all(df, target, train_a=1, train_par=list(range(0,50)), test_par=list(range(50,75))):
    import numpy as np
    import pandas as pd
    
    #Select proper participants for training
    df_train = df.loc[df['PID'].isin(train_par)]
    #Select only musical sections
    df_train = df_train.iloc[(int(len(df_train)/2)):,:]
    #columns to drop from X
    drop_cols = ['PID', 'SID', 'target_1_sec', 'X', 'Y','Z', 'segment']
    #set milliseconds as index
    df_train = df_train.set_index('millisecond')
 
    #
    train_X = df_train.drop(columns=drop_cols)#.to_numpy()
    train_y = df_train[target]#.to_numpy()
    
    #Select proper participants for training
    df_test = df.loc[df['PID'].isin(test_par)]
    #Select only musical sections
    df_test = df_test.iloc[(int(len(df_train)/2)):,:]
    #columns to drop from X
    drop_cols = ['PID', 'SID', 'target_1_sec', 'X', 'Y','Z', 'segment']
    #set milliseconds as index
    df_test = df_test.set_index('millisecond')
    
    
    test_X = df_test.drop(columns=drop_cols)
    test_y = df_test.loc[:,[target,'PID']]
        
    return train_X, test_X, train_y, test_y, train_par, test_par
