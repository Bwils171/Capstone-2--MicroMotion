def score_model(model, participants, test_y, test_X, log, model_name='Model'):
    #create dataframe of scores, predictions and actual values for visualization and comparison
    
    #import necessary pacakges
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_percentage_error
    import pandas as pd
    import numpy as np
    
    #Create dictionary for visualization
    par_scores = {'par':[],'pred':[], 'true':[],'MAPE':[]}
    
    #cycle through test participants
    for par in participants['test']:
        #Create mask using PID values
        mask = test_y.reset_index(drop=True)['PID']==par
        #Make predictions
        pred_X = test_X[mask,:]
        y_pred = model.predict(pred_X)
        
        #save true values, participant number, predicted values
        y_true = test_y.loc[test_y['PID']==par, 'target_1_sec'].to_numpy()
        par_scores['par'].append(par)
        par_scores['pred'].append(y_pred)
        par_scores['true'].append(y_true)
        
        #score using mean absolute error and mean absolute percentage error
        MAPE = mean_absolute_percentage_error(y_true, y_pred)
        MAE = mean_absolute_error(y_true, y_pred)
        par_scores['MAPE'].append(MAPE)
        
        #Add data to log
        log['par'].append(par)
        log['MAPE'].append(MAPE)
        log['MAE'].append(MAE)
        log['model'].append(model_name)
        for key, value in model.get_params().items():
            log[key].append(value)
        
        #convert dictionary to dataframe
        par_scores_df = pd.DataFrame(par_scores)   
    
    #output updated log and dataframe for visualization
    return par_scores_df, log    
