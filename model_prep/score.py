def score_model(model, participants, test_y, test_X, log, model_name='Model'):
    
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_percentage_error
    import pandas as pd
    import numpy as np
    
    par_scores = {'par':[],'pred':[], 'true':[],'MAPE':[]}
    for par in participants['test']:
        mask = test_y.reset_index(drop=True)['PID']==par
        pred_X = test_X[mask,:]
        y_pred = model.predict(pred_X)
        y_true = test_y.loc[test_y['PID']==par, 'target_1_sec'].to_numpy()
        par_scores['par'].append(par)
        par_scores['pred'].append(y_pred)
        par_scores['true'].append(y_true)
    
        MAPE = mean_absolute_percentage_error(y_true, y_pred)
        MAE = mean_absolute_error(y_true, y_pred)
        par_scores['MAPE'].append(MAPE)
        
        log['par'].append(par)
        log['MAPE'].append(MAPE)
        log['MAE'].append(MAE)
        log['model'].append(model_name)
        for key, value in model.get_params().items():
            log[key].append(value)
        
        
        par_scores_df = pd.DataFrame(par_scores)   
    return par_scores_df, log    
