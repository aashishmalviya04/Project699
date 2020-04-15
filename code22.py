#After the successful installation of IDE, we import  required libraries for proper program execution of prediction model, following are the codes written for libraries:

import  pandas as pd
import numpy as np 
from  sklearn.model_selection import train_test_split 
from  sklearn.model_selection import cross_val_score 
from  sklearn.model_selection import GridSearchCV 
from  sklearn.preprocessing import StandardScaler 
from  sklearn.pipeline import  Pipeline 
from  sklearn.linear_model import LinearRegression
from  sklearn.linear_model import  Ridge 
from  sklearn.linear_model import  Las 
from  sklearn.linear_model import ElasticNet 
from  sklearn.tree import DecisionTreeRegressor 
from  sklearn.ensemble import RandomForestRegressor 
from  sklearn.svm import  SVR 
from  sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt 
from  sklearn import  preprocessing 
from  sklearn.preprocessing import MinMaxScaler
from  time import  time 
from  scipy.stats import  randint as sp_randint 




#Following program has been written to train the model:

train_data1 = pd.read_csv(‘Scada.csv’,parse_dates={‘dt’ : [‘date’, ‘time’]},sep=”;”,infer_datetime_format=True,low_memory=False, index_col=’dt’)
train_data_temp = pd.read_csv(‘temp complete.csv’,parse_dates={‘dt’ : [‘date’, ‘time’]},sep=”;”,infer_datetime_format=True,low_memory=False, index_col=’dt’) 
siz=len(train_dt_temp)
train_data = train_data1 [“dt_tm”].resample(‘20T’).sum()[0:siz] tem= train_dt_temp [“tem”].resample(“20T”).median()
hmd= train_dt_temp [“hmd”].resample(“20T”).median() win_spd = train_dt_temp [“win_spd”].resample(“20T”).median() egen= train_dt_temp [“egen”].resample(“20T”).median()
elec_con = train_dt_temp [“elec_con”].resample(“20T”).median()

dt=pd.DataFrame({
“month”: train_data.index.month, “day”: train_data.index.day, “hour”: train_data.index.hour, 
“minute”: train_data.index.minute,
“dayofyear”:train_data.index.dayofyear”weekofyear”:train_data.index.weekofyear,
“dayofweek”:train_data.index.dayofweek,
“quarter”: train_data.index.quarter, 
“tem “:tem,
“hmd”:hmd,
“win_spd“: win_spd, “egen”:egen,
“elec_con“: elec_con,
“dt_tm“: train_data
})



fig_size = plt.rcParams[“figure.figsize”] fig_size[0] = 20
fig_size[1] = 6 dff.fillna(0, inplace=True) dataset =dff.values
max_val=dataset[:,0].max() min_val=dataset[:,0].min() print(max_val) print(min_val) size=len(dataset) xt=int(size*0.75)
print(xt) xy=size-xt
scaler = MinMaxScaler(feature_range=(0, 1)) scaler = scaler.fit(dataset)
normalized = scaler.transform(dataset) X_train = normalized[0:xt, 1:]
Y_train = normalized[0:xt, 0] X_test = normalized[xt:, 1:] Y_test = normalized[xt:, 0]

#Linear Regression Model:

from  sklearn.linear_model import  LinearRegression
for in range(1,2,1):
    linear_reg =LinearRegression()
    reg_scaled = linear_reg.fit(X_train, Y_train)
    y_train _scaled_fit = reg_scaled.predict(X_train)
    real_val = Y_train * (max_val – min_val) + min_val
    predict_val = y_train_scaled_fit * (max_val – min_val) + min_val
    y_test_scaled_fit = reg_scaled.predict(X_test) 
    realvaluet = Y_test * (max_val – min_val) + min_val
    predicvaluestf = Y_test_scaled_fit * (max_val – min_val) + min_val
    print(“rmsesamp, “+ str(np.round(np.sqrt(mean_squared_error(real_val,predict_val)), 4))+” ,rmse test, “+str(np.round(np.sqrt(mean_squared_error(realvaluet,predicvaluestf)), 4))+”,mae train, “+ str(np.round(mean_absolute_error(real_val,predict_val), 4))+” ,mae test, “+str(np.round(mean_absolute_error(realvaluet,predicvaluestf), 4))+”,r2 train, “+ str(np.round(r2_score(real_val,predict_val), 4))+” ,r2 test, “+str(np.round(r2_score(realvaluet,predicvaluestf), 4)))
    predict=[x for x in predicvaluestf] 
    print(“predict”)
    print(predict) 
    real=[x for x in realvaluet]
    plt.plot(realvaluet[1:3500]) 
    plt.plot(predicvaluestf[1:3500],linestyle=’dashed’,color=’red’) 
    plt.show()


Random Forest:
for I in range(1,10,1):

linear_reg = RandomForestRegressor(max_features= 5, min_samples_split= 10, 
n_estimators= 30, max_depth= None, min_samples_leaf= 10) 
reg_scaled = linear_reg.fit(X_train, Y_train)

y_train_scaled_fit = reg_scaled.predict(X_train) 
realvalue = y_train * (max_val–min_val) + min_val
predict_val = y_train_scaled_fit * (max_val–min_val) + min_val 
y_test_scaled_fit = reg_scaled.predict(X_test) 
realvaluet = y_test * (max_val–min_val) + min_val
predicvaluestf = y_test_scaled_fit * (max_val–min_val) + min_val
print(“rmse train, “+ str(np.round(np.sqrt(mean_squared_error(real_val,predict_val)), 4))+” ,rmse test, “+str(np.round(np.sqrt(mean_squared_error(realvaluet,predicvaluestf)), 4))+”,mae train, “+ str(np.round(mean_absolute_error(real_val,predict_val), 4))+” ,mae test, “+str(np.round(mean_absolute_error(realvaluet,predicvaluestf), 4))+”,r2 train, “+str(np.round(r2_score(real_val,predict_val), 4))+” ,r2 test, “+str(np.round(r2_score(realvaluet,predicvaluestf), 4)))
predict=[x for x in predicvaluestf] 
print(predict) 
real=[x for x in realvaluet] 
plt.plot(realvaluet[1:500]) 
plt.plot(predicvaluest[1:500],linestyle=’dashed’,color=’red’) 
plt.show()
print(‘---realvalue------‘)
print(real)


#Decision Tree:

for I in range(1,10,1): 
linear_reg = DecisionTreeRegressor(max_features= 5, min_samples_split=3, max_depth= None, min_samples_leaf= 3)
reg_scaled = linear_reg.fit(X_train, y_train) 
y_train _scaled_fit = reg_scaled.predict(X_train)
real_val = y_train * (max_val – min_val) + min_val
predict_val = y_train_scaled_fit * (max_val – min_val) + min_val
y_test_scaled_fit = reg_scaled.predict(X_test) 
realvaluet = y_test * (max_val – min_val) + min_val
predicvaluestf = y_test_scaled_fit * (max_val – min_val) + min_val
print(“rmse train, “+ str(np.round(np.sqrt(mean_squared_error(real_val, predict_val)), 4))+” ,rmse test, “+str(np.round(np.sqrt(mean_squared_error(realvaluet,predicvaluestf)), 4))+”,mae train, “+ str(np.round(mean_absolute_error(real_val,predict_val), 4))+” ,mae test, “+str(np.round(mean_absolute_error(realvaluet,predicvaluestf), 4))+”,r2 train, “+str(np.round(r2_score(real_val,predict_val), 4))+” ,r2 test, “+str(np.round(r2_score(realvaluet,predicvaluestf), 4)))

predict=[x for x in predicvaluestf] 
print(“predict”)
print(predict)
real=[x for x in realvaluet] 
plt.plot(realvaluet[1:3500])
plt.plot(predicvaluestf[1:3500],
linestyle=’dashed’,color=’red’) 
plt.show()


#After designing the prediction models it is important to calculate parameters and in order to find out the most accurate model we perform grid search on models.

#GridSearch on Decision Tree:

from  sklearn.ensemble import  DecisionTreeRegressor
def  MSE(y_real,y_pred):
    mse = mean_squared_error(y_real, y_pred) print (‘MSE: %2.3f’ % mse)
    returnmse
    
def  R2(y_real,y_pred):
    r_sqr = r2_score(y_real, y_pred) print( ‘R2: %2.3f’ % r2)
    returnr_sqr

def two_score(y_real,y_pred):
    MSE(y_real,y_pred)
    return count 

def two_scorer():
    return make_scorer(two_score, greater_is_better=True)
     
 
 # change it for false if using MSE
clf = DecisionTreeRegressor()


# In order to report best scores, we will be using utility function def report(results, n_top=3):
for I in range(1, n_top + 1):


prtcpnt = np.flatnonzero(results[‘rank_test_score’] == i) for prtcpnt in prtcpnts:
print(“Model with rank: {0}”.format(i))
print(“Mean validation score: {0:.3f} (std: {1:.3f})”.format( results[‘mean_test_score’][ prtcpnt], results[‘std_test_score’][prtcpnt]))
print(“Parameters: {0}”.format(results[‘param’][ prtcpnt])) 
print(“”)


# using full grid search over all parameters param_grid = {“max_depth”: [3,None],
“min_sample_split”: [2, 3,10],
“min_sample_leaf”: [1, 3, 10],
“max_feature”: [2, 3,5,7,9],
}


# run grid search
grid_srch = GridSearchCV(clf, prm_grid=prm_grid) start = time()
grid_srch.fit(X_train, Y_train)
print(“GridSearch CV takes %0.2f sec for %d participates parameter settings.”


% (time() – start, len(grid_search.cv_results_[‘param’]))) report(grid_srch.cv_results_)

												
#GridSearchOn Random Forest:
from sklearn.ensemble import  RandomForestRegressor
def  MSE(y_real,y_pred):
    mse = mean_squared_error(y_real, y_pred) print (‘MSE: %2.3f’ % mse)
    return mse

def  R2(y_real,y_pred):
    r_sqr = r2_score(y_real, y_pred) print( ‘R2: %2.3f’ % r2)
    return r_sqr

def  two_score(y_r,y_uepred):
    MSE(y_real,y_pred) 
    return count 

def two_scorer():
    return make_scorer(two_score, greater_is_better=True) 


# In order to report best scores, we will be using utility function

def  report(results, n_top=3): 
    for I in range(1, n_top + 1):
    prtcpnt = np.flatnonzero(results[‘rank_test_score’] == i) 
    forprtcpnt in prtcpnts:
    print(“Model with rank: {0}”.format(i))
    print(“Mean validation score: {0:.3f} (std: {1:.3f})”.format( results[‘mean_test_score’][ prtcpnt], results[‘std_test_score’][ prtcpnt]))
    print(“Parameters: {0}”.format(results[‘param’][ prtcpnt])) 
    print(“”)



# using full grid over all parameters
param_grid = {“max_depth”: [3, None],
“n_estimators”:[10,20,30,50,100,200,300],
“max_features”: [2, 3,5,7,9],
“min_samples_split”: [2, 3, 10],
“min_samples_leaf”: [1, 3, 10],
 #”bootstrap”: [True, False],
}


# run grid search
grid_srch = GridSearchCV(clf, prm_grid=prm_grid) start = time()
grid_srch.fit(X_train, Y_train)
print(“GridSearch CV takes %0.2 f sec for %d participants parameter settings.”
% (time() – start, len(grid_search.cv_results_[‘params’])))
