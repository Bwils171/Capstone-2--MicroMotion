def plot_predictions(results, participants, model_name='Model'):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    ind = 0
    for par in participants['test']:
        prediction = results.loc[results['par']==par, 'pred'][ind]
        truth = results.loc[results['par']==par, 'true'][ind]
        compare = pd.DataFrame({'Truth':truth, 'Prediction':prediction})
        compare.rolling(100).mean().plot(figsize=(20,6))
        plt.title('{model} Comparision of Actual to Proposed Participant {par}'.format(model=model_name, par=par))
        plt.xlabel('Time (tenths of a second)')
        plt.ylabel('Movement (mm)')
        plt.text(x=1750, y=5.5, s='MAPE = {}%'.format(int(results['MAPE'][ind]*100)), fontsize='x-large')
        plt.ylim(1,7)
        plt.show()
        ind +=1
