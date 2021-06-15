def plot_predictions(results, participants, model_name='Model'):
    #visualize predictions for each individual participant 
    
    #import packages
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    #set index to zero
    ind = 0
    
    #cycle through test participants
    for par in participants['test']:
        #Extract predictions and truths from dataframe and combine for plotting
        prediction = results.loc[results['par']==par, 'pred'][ind]
        truth = results.loc[results['par']==par, 'true'][ind]
        compare = pd.DataFrame({'Truth':truth, 'Prediction':prediction})
        
        #plot rolling 10 second mean of predictions and truths 
        compare.rolling(100).mean().plot(figsize=(20,6))
        plt.title('{model} Comparision of Actual to Proposed Participant {par}'.format(model=model_name, par=par))
        plt.xlabel('Time (tenths of a second)')
        plt.ylabel('Movement (mm)')
        plt.text(x=1750, y=7.0, s='MAPE = {}%'.format(int(results['MAPE'][ind]*100)), fontsize='x-large')
        plt.ylim(2.5,15)
        plt.show()
        ind +=1
