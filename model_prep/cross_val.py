from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split, RandomizedSearchCV




def cross_val_time(model, algo,  X, y, split=5):
    #splits into ordered folds that can to allow for time series cross validation and reporting of scores
    count=1
    tscv = TimeSeriesSplit(n_splits=split)
    scoretemp = {algo+'_MAE':[], algo+'_MSE':[], algo+'_MPE':[]}
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(train_index[0], train_index[-1], test_index[0], test_index[-1])
        with joblib.parallel_backend('dask'):
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scoretemp[algo + '_MAE'].append(mean_absolute_error(y_test, y_pred))
        scoretemp[algo + '_MSE'].append(mean_squared_error(y_test, y_pred))
        scoretemp[algo + '_MPE'].append(mean_absolute_percentage_error(y_test, y_pred))
        for i in scoretemp.keys():
            print(i + ': '+ str(scoretemp[i][count-1]))
        count+=1
    scoretemp = pd.DataFrame(scoretemp)
    return scoretemp