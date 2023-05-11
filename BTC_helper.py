# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 23:43:09 2021

@author: bhask
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:37:34 2021

@author: bhask
Keras Attention  :https://github.com/philipperemy/keras-attention-mechanism
#https://www.programcreek.com/python/example/89676/keras.layers.Conv1D
"""

from pandas import Series
import pandas as pd ,numpy as np ,matplotlib.pyplot as plt 
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler,StandardScaler
from scipy.signal import savgol_filter

#from bayes_opt import BayesianOptimization
#from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy as bp
from pandas import read_csv
import matplotlib.pyplot as pyplot
from sklearn import metrics # for model evalution
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
from scipy import stats 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# Model imports 
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Flaregultten
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation,Bidirectional
from tensorflow.keras import layers
from tensorflow.keras.models import load_model,Model
from tensorflow.keras import backend
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.activations import elu
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger
from tensorflow.keras.models import Model
from attention import Attention
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LeakyReLU

#Feature Selection
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from hampel import hampel  


#Model Optimization
#from bayes_opt import BayesianOptimization
#from PyEMD import EMD,EEMD, Visualisation
#import configparser # https://docs.python.org/3/library/configparser.html

# In[1]:----------------------------------UTILITY FUNCTIONS-----------------------------------------------------


def cubic_spline_interpolation(df):
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    # df = df[~df.index.duplicated()] #Remove duplicate index (if any)
    # df["Index"] = df["Index"].astype(str)
    # df.set_index("Index", inplace=True)
    df.index = pd.to_datetime(df.index) #Convert index to datetime
    df = df.iloc[:,-1:].resample("D").interpolate(method='spline', order=2)
    df = df.reset_index()
    
    return df

def plot_full_chart_true_pred(yhat_test_inv, test_y_orig,df_reduced_corr):
    # Create a DataFrame of Real and Predicted values
    true_vs_pred = pd.DataFrame({
        "Real": test_y_orig.ravel(),
        "Predicted": yhat_test_inv.ravel()
    }, index = df_reduced_corr.index[-len(test_y_orig): ]) 
    true_vs_pred.head()
    plt.rcParams["figure.figsize"] = [13,6]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300
    true_vs_pred.plot()
    plt.show()
    return true_vs_pred

from hampel import hampel  
def hampel_filter_data(data, columns,win_size=15,isImputable=True):
    for col in columns: 
        if(isImputable ==False):
            outlier_indices = hampel(ts = data[col], window_size = win_size,imputation=isImputable)
            filtered_d = data[col].drop(outlier_indices)
            data[col]=filtered_d
        else:
            data[col] = hampel(data[col], win_size,3,isImputable)
    return data

def savgol_filter_data(data, columns,window_size=21, polynomial_degree=3):
    for col in columns:
        data[col]= savgol_filter(data[col], window_size, polynomial_degree,mode='nearest')
    return data

def scale_data(data, columns, scaler):
    for col in columns:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data,scaler

def transform_data(data, columns, scaler):
    for col in columns:
        scaler.fit(data[col].values.reshape(-1, 1))
        data[col] = scaler.transform(data[col].values.reshape(-1, 1))
    return data,scaler

def inverse_scale_data(inv_data, columns, scaler):
    for col in columns:
        #data[col] = scaler.inverse(data[col].values.reshape(-1, 1))
        inv_data[col] = scaler.inverse(inv_data[col].values)
    return inv_data

#Plot True and Predicted values
def chart_regression(test_y_orig,y_pred):
    fig, ax = plt.subplots(figsize=(13,6), dpi=300)
    plt.plot(test_y_orig, label='True') #Expected values
    plt.plot(y_pred, label='Predicted') #Predicted values
    plt.ylabel('Exchange Prices (USD)', fontsize=8)
    plt.legend()
    plt.show()
# Loss values are calculated for every training epoch and are visualized
def plot_loss(history, title):
  """function that plots the loss results of the model"""
  plt.figure(figsize=(8,6),dpi=300)
  plt.plot(history.history['loss'], 'o-', mfc='none', markersize=3, 
  label='Train')
  plt.plot(history.history['val_loss'], 'o-', mfc='none', markersize=3, label='Test')
  plt.title(title)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
  
def plot_validation_accuracy(history, title,error,val_error):
  """function that plots the loss results of the model"""
  plt.figure(figsize=(8,6),dpi=300)
  plt.plot(history.history[error], 'o-', mfc='none', markersize=3, 
  label='Train')
  plt.plot(history.history[val_error], 'o-', mfc='none', markersize=3, label='Test')
  plt.title('LSTM Model RMSE')
  plt.xlabel('Epoch')
  plt.ylabel('Root Mean Square Error')
  plt.legend()
  plt.show()
# re-frame this time series dataset as a supervised learning problem with a window width of one/given width to predict the next time step (t+1)
# use lagged observations (e.g. t-1) as input variables to forecast the current time step (t)
# reframed dataframe will be (t-1) as X(input predictors) and t as y()
#the current time (t) and future times (t+1, t+n) are forecast times and past observations (t-1, t-n) are used to make forecasts
#One-Step Forecast: This is where the next time step (t+1) is predicted.
#Multi-Step Forecast: This is where two or more future time steps are to be predicted.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# Here is created input columns which are (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# Here is created output/forecast column which are (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def split_series(series, n_past, n_future):
  # n_past ==> no of past observations
  # n_future ==> no of future observations 
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)

def create_dataset(dataset,features, look_back=1):
	dataset = np.insert(dataset,[0]*look_back,0)    
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	dataY= np.array(dataY)        
	dataY = np.reshape(dataY,(dataY.shape[0],features))
	dataset = np.concatenate((dataX,dataY),axis=1)  
	return dataset
#Plot the historical trend (300dpi resolution) - Execute this function after splitting the data. 
#In this case the data doesn't needs to be scaled by the minmaxscaler
def line_plot(train, test, label1=None, label2=None, title='', lw=2,y_axis_title=''):
    fig, ax = plt.subplots(1, figsize=(13, 7),dpi = 300)
    ax.plot(train, label=label1, linewidth=lw)
    #ax.plot(test, label=label2, linewidth=lw)
    ax.plot([None for i in train] + [x for x in test],label=label2, linewidth=lw)
    ax.set_ylabel(y_axis_title, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    
# def train_test_split(df, test_size=0.1):
#     # split_row = len(df) - int(test_size * len(df))
#     # train_data = df[:split_row]
#     # test_data = df[split_row:]
#     values=df
#     train_size = int(len(values)*test_size)
#     train_data, test_data= values[0:train_size], values[train_size:len(values)]
#     return train_data, test_data

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mape(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


def timeseries_evaluation_metrics_func(y_true, y_pred):
    print('Evaluation metric results:-')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mape(y_true, y_pred)}')
    #print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))



# Apply Feature Selection
def apply_feature_selection_RF_Boruta(raw_dataset,max_itr):
    cols = [i for i in range(0,len(raw_dataset.columns)-1)] 
    select_cols = [i for i in range(len(raw_dataset.columns))] 
    last_column = select_cols[len(select_cols)-1:len(select_cols)]
    X_feat_sel=raw_dataset.iloc[:, cols].values
    y_feat_sel=raw_dataset.iloc[:,(last_column)].values
    rf_regressor_model = RandomForestRegressor(n_jobs= 4,oob_score= True)
    feat_selector = bp(rf_regressor_model,n_estimators = 'auto', verbose= 2,max_iter= max_itr)
    feat_selector.fit(X_feat_sel, y_feat_sel)
    selected_features = [raw_dataset.columns[i] for i, x in enumerate(feat_selector.ranking_) if x]
    selected_RF_features = pd.DataFrame({'Feature':list(selected_features),'Ranking':list(feat_selector.support_)})
    selected_RF_features = selected_RF_features[selected_RF_features.Feature != 'Date']
    selected_RF_features.sort_values(by='Ranking')
    # dataset_feature_names=list(dataset.columns) # x+y column names
    # dataset_feature_names.remove('df_BTCdata_Close') # remove the target(y)
    #X_filtered = feat_selector.transform(X)
    selected_features = selected_RF_features[selected_RF_features['Ranking'] != True]
    cols_to_drop=[]
    for vals in selected_features.Feature:
        cols_to_drop.append(vals)
    #print(cols_to_drop)    
    df_result=raw_dataset.drop(columns=cols_to_drop)
    #df_result.info()
    print("==============BORUTA==============")
    print ("No. of confirmed features",feat_selector.n_features_)
    print (feat_selector.support_)
    print("Features that fall into the Acceptance area : ",feat_selector.ranking_)
    print("Impurity-based feature importances of the forest - ",selected_RF_features)
    return df_result

def apply_min_max_scaling(train_df, test_df ):
    scalers={}
    for i in train_df.columns:
        scaler = MinMaxScaler(feature_range=(-1,1))
        s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+ i] = scaler
        train[i]=s_s
        #Scale Test values
        test = test_df
    for i in train_df.columns:
        scaler = scalers['scaler_'+i]
        s_s = scaler.transform(test[i].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+i] = scaler
        test[i]=s_s
    return train_df, test_df,scalers 

# Here was prepared column for visualizing
#Visualize the trend of each column
def plot_each_column_trend(raw_dataset):
    print("Visualizing the trend of each column")
    col_idx=[]
    for cnt in range(0,len(raw_dataset.columns)):
        col_idx.append(cnt)
    groups = col_idx
    i = 1
    # plot each column
    plt.figure(figsize=(19,62))
    for group in groups:
     	pyplot.subplot(len(groups), 1, i)
     	pyplot.plot(raw_dataset.values[:, group])
     	pyplot.title(raw_dataset.columns[group], y=0.5, loc='right')
     	i += 1
    pyplot.show()
    print("Done !")    

# Join all dataframes provided in a list
def join_all_dataframes(df_Tech_Ind_raw):
    count_ind=0    
    df_Tech_Ind_final = pd.DataFrame() # begin empty
    for indicator in df_Tech_Ind_raw:
        print(count_ind)
        if count_ind==0:
            df_Tech_Ind_final =indicator #If it is the first index, store the Close price against which joins would be made
        else:
            df_Tech_Ind_final = df_Tech_Ind_final.join(indicator,how='left')
        count_ind+=1
    return df_Tech_Ind_final


#function for listing vif values
def vif_values(X):
    add_constant(X)
    df=pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    df= pd.Series(df, dtype=np.float64, name='VIF')
    #df.sort_values(by='VIF')
    return df.to_frame()

# Find correlated features without keeping specified columns
def correlation(dataset, threshold):
    #Usage : corr_features = correlation(df, 0.9)
    #print('correlated features: ', len(set(corr_features)) )
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

# define base model
def create_model(neuron_units1,neuron_units2,time_steps_lookback,n_features,dropout_rate,recurrent_dropout,output_labels,optimizer_name,activation_func1,activation_func2,loss_func,metric):
    model = Sequential()
    model.add(LSTM(neuron_units1,activation=activation_func1, return_sequences=True,input_shape=(time_steps_lookback, n_features))) # Hidden 1-The LSTM input layer is defined by the input_shape argument on the first hidden layer.
    model.add(Dropout(dropout_rate)) 
    model.add(Dropout(recurrent_dropout)) 
    model.add(LSTM(neuron_units2,activation=activation_func2,return_sequences=True)) # Hidden 2
    model.add(Dropout(dropout_rate)) 
    model.add(Dropout(dropout_rate)) 
    model.add(Dropout(recurrent_dropout)) 
    model.add(Dense(units=output_labels)) # Output Layer 4
    model.compile(loss=loss_func, optimizer=optimizer_name,metrics=metric)
    model.summary()
    return model
def create_model2(neuron_units1,neuron_units2,time_steps_lookback,n_features,dropout_rate,recurrent_dropout,output_labels,optimizer_name,activation_func1,activation_func2,loss_func,metric):
    model = Sequential()
    model.add(LSTM(neuron_units1,activation=activation_func1, return_sequences=True,input_shape=(time_steps_lookback, n_features))) # Hidden 1-The LSTM input layer is defined by the input_shape argument on the first hidden layer.
    model.add(Dropout(dropout_rate)) 
    model.add(Dropout(recurrent_dropout)) 
    # model.add(LSTM(neuron_units2,activation=activation_func2,return_sequences=True)) # Hidden 2
    # model.add(Dropout(dropout_rate)) 
    model.add(Dropout(dropout_rate)) 
    model.add(Dropout(recurrent_dropout)) 
    model.add(Dense(units=output_labels)) # Output Layer 4
    model.compile(loss=loss_func, optimizer=optimizer_name,metrics=metric)
    model.summary()
    return model

# Reusable model fit wrapper.
def epocher(batch_size):
    # Create the print weights callback.
    from keras.callbacks import LambdaCallback
    history = []
    print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: history.append(clf.layers[0].get_weights()))
    

#Add Model Checkpoints
def run_and_fit_model(model,tr_X, tr_y,val_X_var,val_y_var,batch_size,epoch_count,ReduceLR_factor,early_stopping_patience,lr_patience):
    filepath="saved_models/LSTM_improv_{epoch:02d}.hd5" #File name includes epochand val accuracy
    mcp_save= ModelCheckpoint(filepath,save_best_only=True, monitor='val_loss', mode='min') 
    early_stop = EarlyStopping(monitor='val_loss', patience= early_stopping_patience, verbose=0, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=ReduceLR_factor, patience=lr_patience, verbose=1, min_delta=1e-4, mode='min') #reduce learning rate when model is not improving
    log_csv=CSVLogger('saved_models/my_logs.csv',separator=',',append=False)
    callbacks_list=[early_stop,reduce_lr_loss] #Only early stop required for initial experiments
    history = model.fit(tr_X, tr_y , epochs=epoch_count, batch_size=batch_size,validation_data=(val_X_var,val_y_var),verbose=2, callbacks=callbacks_list,use_multiprocessing=True, workers=8)
    return history

def run_and_fit_classification_model(model,tr_X, tr_y,val_X,val_y,batch_size,epoch_count,ReduceLR_factor,early_stopping_patience,lr_patience):
    filepath="saved_models/LSTM_improv_{epoch:02d}.hd5" #File name includes epochand val accuracy
    mcp_save= ModelCheckpoint(filepath,save_best_only=True, monitor='val_loss', mode='max') 
    early_stop = EarlyStopping(monitor='val_loss', patience= early_stopping_patience, verbose=0, mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=ReduceLR_factor, patience=lr_patience, verbose=1, min_delta=1e-4, mode='min') #reduce learning rate when model is not improving
    log_csv=CSVLogger('saved_models/my_logs.csv',separator=',',append=False)
    callbacks_list=[early_stop,reduce_lr_loss] #Only early stop required for initial experiments
    history = model.fit(tr_X, tr_y , epochs=epoch_count, batch_size=batch_size,validation_data=(val_X , val_y),verbose=2, callbacks=callbacks_list,use_multiprocessing=True, workers=8)
    return history

def run_and_fit_hybrid_model(model,tr_X, tr_y,val_X,val_y,batch_size,epoch_count,ReduceLR_factor,early_stopping_patience,lr_patience):
    filepath="saved_models/LSTM_improv_{epoch:02d}.hd5" #File name includes epochand val accuracy
    mcp_save= ModelCheckpoint(filepath,save_best_only=True, monitor='val_loss', mode='max') 
    early_stop = EarlyStopping(monitor='val_loss', patience= early_stopping_patience, verbose=0, mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=ReduceLR_factor, patience=lr_patience, verbose=1, min_delta=1e-4, mode='min') #reduce learning rate when model is not improving
    log_csv=CSVLogger('saved_models/my_logs.csv',separator=',',append=False)
    callbacks_list=[early_stop,reduce_lr_loss] #Only early stop required for initial experiments
    history = model.fit(tr_X, tr_y , epochs=epoch_count, batch_size=batch_size,validation_data=(val_X , val_y),verbose=2, callbacks=callbacks_list,use_multiprocessing=True, workers=8)
    return history

def exponential_smoothing(series, alpha):
    """given a series and alpha, return series of expoentially smoothed points"""
    # http://ethen8181.github.io/machine-learning/time_series/1_exponential_smoothing.html
    results = np.zeros_like(series)

    # first value remains the same as series,
    # as there is no history to learn from
    results[0] = series[0] 
    for t in range(1, series.shape[0]):
        results[t] = alpha * series[t] + (1 - alpha) * results[t - 1]

    return results

def plot_exponential_smoothing(series, alphas):
    """Plots exponential smoothing with different alphas."""  
    plt.figure(figsize=(15, 7),dpi=300)
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label='Alpha {}'.format(alpha))

    plt.plot(series, label='Actual')
    plt.legend(loc='best')
    plt.axis('tight')
    plt.title('Exponential Smoothing')
    plt.grid(True)
# In[1]:-------------------------------LOAD AND MERGE TECHNICAL AND FUNDAMENTAL PREDICTORS--------------------------------------------------------
def load_Tech_Indicators_Raw():
    #df_Tech_Ind= read_csv("C:/Users/bhask/OneDrive/Desktop/Paper II/Code/OpenSwarm/ML Learning Experiments/BTC_data/Technical_indicators_FS_26_Oct.csv", header=0, index_col=0)
    df_Tech_Ind= read_csv("C:/Users/bhask/OneDrive/Desktop/Paper II/Code/OpenSwarm/ML Learning Experiments/BTC_data/Technical_indicators_raw_9_Nov.csv", header=0, index_col=0)
    print("Checking missing records in both datasets..")
    num_missing=df_Tech_Ind.isnull().sum()
    print(num_missing[num_missing >0]) #Check missing records in Technical Indicators
    print("Done!")
    df_Tech_Ind['close'].sum()
    return df_Tech_Ind

def load_fundamental_indicators_raw():
    df_fundamental = read_csv('C:/Users/bhask/OneDrive/Desktop/Paper II/Code/OpenSwarm/ML Learning Experiments/BTC_data/Bitcoin_all_intervals_14_11_2021.csv', header=0, index_col=0)
    print("Checking missing records in both datasets..")
    num_missing=df_fundamental.isnull().sum()
    print(num_missing[num_missing >0]) #Check missing records in Technical Indicators
    print("Done!")
    df_fundamental['df_BTCdata_Close'].sum()
    return df_fundamental
    

def load_data():
    df_fundamental = read_csv('C:/Users/bhask/OneDrive/Desktop/Paper II/Code/OpenSwarm/ML Learning Experiments/BTC_data/Bitcoin_all_intervals_02_09_2021.csv', header=0, index_col=0)
    df_Tech_Ind= read_csv("C:/Users/bhask/OneDrive/Desktop/Paper II/Code/OpenSwarm/ML Learning Experiments/BTC_data/Technical_indicators_raw.csv", header=0, index_col=0)
    #Technical Indicators and fundamental indicators hsould be of same date
    df_Tech_Ind=df_Tech_Ind[(df_Tech_Ind.index.get_level_values(0) >= '2016-01-01') & (df_Tech_Ind.index.get_level_values(0) <= '2021-08-29')]
    #Both fundamental and tech indicators should have same date format
    df_Tech_Ind.index = pd.to_datetime(df_Tech_Ind.index, format = '%Y-%m-%d').strftime('%Y-%m-%d')
    df_fundamental.index= pd.to_datetime(df_fundamental.index, format = '%m/%d/%Y').strftime('%Y-%m-%d')
    print("Checking missing records in both datasets..")
    num_missing=df_Tech_Ind.isnull().sum()
    print(num_missing[num_missing >0]) #Check missing records in Technical Indicators
    num_missing=df_fundamental.isnull().sum()
    print(num_missing[num_missing >0]) #Check missing records in Fundamental Indicators
    print("Done!")
    df_Tech_Ind['close'].sum()
    df_fundamental['df_BTCdata_Close'].sum()
    data_frames_list = [df_Tech_Ind, df_fundamental] # compile the list of dataframes to be merged
    df_tech_fundamental = join_all_dataframes(data_frames_list) #Merge both Tech and Fundamental indicators in a new dataframe
    num_missing=df_tech_fundamental.isnull().sum() #Check missing records
    #Apply Interpolation on missing records 
    print("Null records..")
    print(num_missing[num_missing >0]) #Check missing records in merged dataframe
    print("Applying interpolation..")
    df_tech_fundamental = df_tech_fundamental.fillna(method='bfill')
    df_tech_fundamental = df_tech_fundamental.fillna(method='ffill')
    print("Done !")
    num_missing=df_tech_fundamental.isnull().sum()
    print(num_missing[num_missing >0]) #Check missing records in merged dataframe
    df_tech_fundamental=df_tech_fundamental.drop(['close'],axis=1)
    df_tech_fundamental.info()
    return df_tech_fundamental

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[:,0][-interval]

def split_data_train_val_test(df,train_size=0.80,val_size=0.10,test_size=0.10):
    n = len(df)
    train_df = df[0:int(n*train_size)]
    val_df = df[int(n*train_size):int(n*0.9)]
    #val_df = df[int(n*train_size):int(2000)]
    test_df = df[int(n*0.9):]
    return train_df ,val_df,test_df

def split_data_train_test(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_df = df[:split_row]
    test_df = df[split_row:]
    return train_df, test_df

def split_train_val_test_shuffle(X,y,ratio_train = 0.8,ratio_val = 0.1,ratio_test = 0.1): # Defines ratios, w.r.t. whole dataset.
    '''Usage: 
        X_train1,X_test1,X_validation1,y_validation1,y_train1,y_test1 = split_train_validation_test(X,y,.8,0.1,0.1)
    '''
    # Test split
    X_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=ratio_test)
    
    # Adjust value ratio for the remaining dataset.
    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining
    
    # Produces train and val splits.
    X_train, X_validation, y_train, y_validation = train_test_split(X_remaining, y_remaining, test_size=ratio_val_adjusted,shuffle=False,random_state=1)
    return X_train,X_test,X_validation,y_validation,y_train,y_test

# def svm_baseline():
    
#     train_df = (train_df - train_mean) / train_std
#     val_df = (val_df - train_mean) / train_std
#     test_df = (test_df - train_mean) / train_std

#     training_data, validation_data, test_data = mnist_loader.load_data()
#     # train
#     regr = SVR()
#     regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
#     regr.fit(X, y)
#     clf.fit(training_data[0], training_data[1])
#     # test
#     predictions = [int(a) for a in regr.predict(test_data[0])]
#     num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
#     print "Baseline classifier using an SVM."
#     print "%s of %s values correct." % (num_correct, len(test_data[1]))

def feature_selection_TI(df_tech_indicators):
        df_tech_fund_reduced=df_tech_indicators[['10 period FISH.', 'ROC', 'MSD', 'MOM','close' ]]
        #df_tech_fund_reduced['close'] = exponential_smoothing(df_tech_fund_reduced['close'].values,.3)
        return df_tech_fund_reduced

def feature_selection_TI_Interval1(df_tech_indicators):
    # df_tech_fund_reduced=df_tech_indicators[['10 period FISH.', 'ROC', 'MSD', 'MOM','close' ]]
    # df_tech_fund_reduced['close'] = exponential_smoothing(df_tech_fund_reduced['close'].values,.3)
    return df_tech_fund_reduced

def feature_selection_fundamental(df_fundamental):
     df_fund_reduced=df_fundamental[['NETWORK_ACTIVITY_output_value_per_day', 'NETWORK_ACTIVITY_transactions_excluding_popular_adresses','MINING_INFORMATION_total_transaction_fees_BTC','NETWORK_ACTIVITY_mempool_size_bytes',
'MINING_INFORMATION_fees_per_transaction_USD','MARKET_SIGNALS_network_value_to_transactions','MINING_INFORMATION_total_transaction_fees_USD','MINING_INFORMATION_miners_revenue_USD',
'google_trends_index_BTC','df_BTCdata_Close']]
     return df_fund_reduced
     

def feature_selection(df_tech_fundamental):
    df_tech_fund_reduced=df_tech_fundamental[['5 period SMA', '10 period SMA', '9 period SMM', '9 period SSMA',
       '9 period EMA', '9 period DEMA', '9 period TEMA', '10 period TRIMA',
       '8 period VAMA', '26 period ZLEMA', '9 period WMA.', '16 period HMA.',
       '9 period EVWMA.', 'VWAP.', 'SMMA', '16 period FRAMA.', 'MOM', 'ROC',
       'SAR', 'BB_MIDDLE', 'MOBO', 'KELTNER_CHANNELS', 'MIDDLE',
       '14 period ADX.', '14 period stochastic RSI.', '10 period FISH.', 'VPT',
       'MSD', 'df_NASDAQ_data_Close', 'df_WTI_crudeoil_data_Close',
       'NETWORK_ACTIVITY_unspent_transaction_outputs',
       'WALLET_ACTIVITY_blockchain.com_wallets', 'df_BTCdata_Close']]

    #High Correlation Treatment
    corr_features = correlation(df_tech_fund_reduced, 0.95)
    print('correlated features: ', len(set(corr_features)) )
    df_reduced_corr= df_tech_fund_reduced.drop(columns=['10 period SMA',
     '10 period TRIMA','16 period FRAMA.','16 period HMA.','26 period ZLEMA','8 period VAMA',
     '9 period DEMA','9 period EMA','9 period EVWMA.','9 period SMM','9 period SSMA',
     '9 period TEMA','9 period WMA.','BB_MIDDLE','KELTNER_CHANNELS','MIDDLE','MOBO',
     'SAR','SMMA','WALLET_ACTIVITY_blockchain.com_wallets'])
    df_reduced_corr.shape
    df_reduced_corr.columns
    
    # df_reduced_corr.shape
    # return df_reduced_corr
    
    # close_price= exponential_smoothing(df_reduced_corr['df_BTCdata_Close'].values, 1.9)
    # df_reduced_corr= df_reduced_corr.assign(df_BTCdata_Close =close_price ) # Add smoothened row to the dataframe
    # df_reduced_corr.columns
    # num_missing=df_reduced_corr.isnull().sum()
    # print(num_missing[num_missing >0])
    
    #Print VIFs after feature selection to check which values to be retained
    vif= vif_values(df_reduced_corr).sort_values(by='VIF')
    filtered_vif_cols=vif.query("VIF < 10") #Filter out columns with VIF less than 10
    filtered_vif_cols.reset_index(level=0, inplace=True) #convert index to column
    for row in filtered_vif_cols['index']:
        print(row)
    df_reduced_vif=df_reduced_corr[['10 period FISH.', 'ROC','MSD','MOM','df_BTCdata_Close']]
    df_reduced_vif.shape
    df_reduced_vif.columns
    #Display all filtered columns with VIFs less than 10
    vif_values(df_reduced_vif).sort_values(by='VIF')
    
    
    df_reduced_vif['df_BTCdata_Close'] = exponential_smoothing(df_reduced_vif['df_BTCdata_Close'].values,.3)
    #df_reduced_vif['df_BTCdata_Close'] = exponential_smoothing(df_reduced_vif['df_BTCdata_Close'].values,.6)
    # df_reduced_vif['10 period FISH.'] = exponential_smoothing(df_reduced_vif['10 period FISH.'].values,1.2)
    # df_reduced_vif['ROC'] = exponential_smoothing(df_reduced_vif['ROC'].values,1.2)
    # df_reduced_vif['MSD'] = exponential_smoothing(df_reduced_vif['MSD'].values,1.2)
    # df_reduced_vif['MOM'] = exponential_smoothing(df_reduced_vif['MOM'].values,1.2)
    
    df_reduced_vif = df_reduced_vif.fillna(method='bfill')
    df_reduced_vif = df_reduced_vif.fillna(method='ffill')

    return df_reduced_vif
global_cols=[]
def feature_selection_evoloutionary(df_tech_fundamental):
    # df=load_data()
    # df_tech_fundamental=df[(df.index.get_level_values(0) >= '2020-01-01') & (df.index.get_level_values(0) <= df.tail(1).index.item())] 
    cols=[]
    cols = df_tech_fundamental.columns[[21, 25, 33, 37,  54, 60, 62, 66, 67,75]]
    global_cols=cols
    df_tech_fund_reduced= df_tech_fundamental[list(cols)]
    
    return df_tech_fund_reduced
    
def make_data_in_LSTM_format(train_reframed,val_reframed,test_reframed,LSTM_NN=True):
    # # Make the value to be predicted(response variable) as the last column 
    # train_reframed = train_reframed[ [ col for col in train_reframed.columns if col != 'var5(t)' ] + ['var5(t)'] ] #var41(t)/ var29(t)
    # val_reframed = val_reframed[ [ col for col in val_reframed.columns if col != 'var5(t)' ] + ['var5(t)'] ]
    # test_reframed = test_reframed[ [ col for col in test_reframed.columns if col != 'var5(t)' ] + ['var5(t)'] ]
    
    test_reframed=test_reframed[list(train_reframed.columns)] #Keep only the selected features
    val_reframed=val_reframed[list(train_reframed.columns)] #Keep only the selected features
    
    train_reframed.columns
    train_reframed.shape,val_reframed.shape, test_reframed.shape
    
    train_X, train_y = train_reframed.values[:, :-1], train_reframed.values[:, -1]
    val_X, val_y = val_reframed.values[:, :-1], val_reframed.values[:, -1]
    test_X, test_y = test_reframed.values[:, :-1], test_reframed.values[:, -1]
    print(train_X.shape, test_y.shape)
    print(val_X.shape, val_y.shape)
    print(test_X.shape, test_y.shape)
    if (LSTM_NN==True):
        # reshape input to be 3D [ a samples, b timesteps, dimensions] for LSTM input format
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1])) # Here 1 is the number of timesteps or lags
        val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        print(train_X.shape, train_y.shape, val_X.shape, val_X.shape,test_X.shape, test_y.shape)
        print("LSTM")
    return train_X, train_y,val_X, val_y,test_X, test_y


# In[1]:--------------------DEVELOP MODELS AND SAVE THE BEST MODEL WITH EARLY STOPPING-------------------------------------------------------------------
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.activations import elu
from tensorflow.keras import backend as K

# In[1]:--------------------Optimization algorithm-------------------------------------------------------------------

# This returns a multi-layer-perceptron model in Keras.
def get_keras_model(num_hidden_layers, 
                    num_neurons_per_layer, 
                    dropout_rate, 
                    activation,train_X):
    # create the MLP model.
    # define the layers.
    inputs = tf.keras.Input(shape=(train_X.shape[1],))  # input layer.
    x = layers.Dropout(dropout_rate)(inputs) # dropout on the weights.
    # Add the hidden layers.
    for i in range(num_hidden_layers):
        x = layers.Dense(num_neurons_per_layer,activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
        #x = Attention(num_neurons_per_layer)(x)
    # output layer.
    #x = Attention(num_neurons_per_layer)(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

#This model is used for the experiments based upon ANN
def get_ANN_attention_model_experiment(num_hidden_layers, num_neurons_per_layer, dropout_rate, activation_func, train_X_var):
    with tf.device('/gpu:0'):
        model_input = tf.keras.Input(shape=(train_X_var.shape[1]))  # input layer.
        for i in range(0,num_hidden_layers):
            if (i==0):
                x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(model_input)
                x = layers.Dropout(dropout_rate)(x)
            else:
                x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(x)
                x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=model_input, outputs=outputs)
        model.summary()
    return model
#Experiment for Stacked Auto Encoder-Decoder
def get_autoencoder_model_experiment(num_hidden_layers, num_neurons_per_layer, dropout_rate, activation_func, train_X_var):
    with tf.device('/gpu:0'):
        model_input = tf.keras.Input(shape=(train_X_var.shape[1]))  # input layer.
        decoder=[]
        for i in range(0,num_hidden_layers):
            if (i==0):
                #Encoder
                encoder = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(model_input)
                encoder = layers.Dropout(dropout_rate)(encoder)
                encoder = Dense(int(num_neurons_per_layer / 2), activation="linear")(encoder)
                encoder = layers.Dropout(dropout_rate)(encoder)
            else:
                # Decoder
                decoder = Dense(int(num_neurons_per_layer/ 2), activation='linear')(encoder)
                decoder = layers.Dropout(dropout_rate)(decoder)
                decoder = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(decoder)
                decoder = layers.Dropout(dropout_rate)(decoder)
        outputs = Dense(1, activation='relu')(decoder)
        #Auto Encoder Decoder model
        model = tf.keras.Model(inputs=model_input, outputs=outputs)
        model.summary()
    return model


def get_CNN_LSTM_Attention_ensemble(num_hidden_layers, num_neurons_per_layer, dropout_rate, activation_func,filters,kernel_size,padding,train_X_var):
    with tf.device('/gpu:0'):
        model_input = (train_X_var.shape[1], train_X_var.shape[2])
        model = Sequential()
        for i in range(0,num_hidden_layers):
            if (i==0):
                model.add(Conv1D(input_shape=model_input ,filters= filters, kernel_size= kernel_size, padding=padding, activation='linear',bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)))
                model.add(MaxPooling1D(pool_size=1))
                model.add(Bidirectional(LSTM(num_hidden_layers,return_sequences=True,activation='linear',bias_regularizer=L1L2(l1=0.0, l2=0.0001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)), input_shape=model_input))
                model.add(Dropout(dropout_rate))
            else:
                #model.add(Bidirectional(LSTM(num_hidden_layers,return_sequences=True,activation='linear',bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))))
                model.add(Dense(num_hidden_layers,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer='he_uniform'))
                model.add(Dropout(dropout_rate))
        model.add(Dense(1,activation='linear'))
        model.summary()
    return model

def get_ANN_attention_model_tech_fundamental(num_hidden_layers, num_neurons_per_layer, dropout_rate, activation_func, train_X_fund_var,train_X_var):
    with tf.device('/gpu:0'):
        model_input_fund = tf.keras.Input(shape=(train_X_fund_var.shape[1]))  # input layer.
        model_input_tech = tf.keras.Input(shape=(train_X_var.shape[1]))  # input layer.
        for i in range(0,num_hidden_layers):
            if (i==0):
                x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(model_input_fund)
                x = layers.Dropout(dropout_rate)(x)
                y = layers.MDense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(model_input_tech)
                y = layers.Dropout(dropout_rate)(y)
            else:
                x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(model_input_fund)
                x = layers.Dropout(dropout_rate)(x)
                y = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(model_input_tech)
                y = layers.Dropout(dropout_rate)(y)
                
        added = layers.Add()([x, y])
        out = keras.layers.Dense(4)(added)
        model = keras.models.Model(inputs=[x, y], outputs=out)
        model.summary()
        # outputs_fund = layers.Dense(1, activation='linear')(x)
        # outputs_tech = layers.Dense(1, activation='linear')(y)
        # model_fund = tf.keras.Model(inputs=model_input_fund, outputs=outputs_fund)
        # model_tech = tf.keras.Model(inputs=model_input_tech, outputs=outputs_tech)
        ## combine the output of the fundamental and technical indicator models
        # combined = tf.keras.layers.Add([model_fund.output, model_tech.output])
        # z = Dense(num_neurons_per_layer, activation="linear", name='dense_pooling')(combined)
        # z = Dense(1, activation="linear", name='dense_out')(z)
        
    return model

def get_ANN_attention_model(num_hidden_layers, num_neurons_per_layer, dropout_rate, activation_func, train_X):
    with tf.device('/gpu:0'):
        model_input = tf.keras.Input(shape=(train_X.shape[1]))  # input layer.
        for i in range(num_hidden_layers):
            #x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(1e-5, 1e-4),activity_regularizer=L1L2(1e-6,1e-4))(model_input)
            #x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(1e-5, 1e-4),activity_regularizer=L1L2(1e-5,1e-4))(model_input)
            
            x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.0001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer='he_uniform')(model_input)
            #x = Bidirectional(LSTM(num_neurons_per_layer, return_sequences=True,activation=activation_func))(model_input)
            x = layers.Dropout(dropout_rate)(x)
            #x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
            print("***********HIDDEN LAYER ",i)
            #x = Attention(num_hidden_layers)(x)
        #x = Dense(1,activation='linear')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=model_input, outputs=outputs)
        model.summary()
    return model

#***********************************THIS MODEL IS USED FOR LSTM EXPERIMENTS***********************************
def get_keras_LSTM_Attention_Model(num_hidden_layers, num_neurons_per_layer, dropout_rate, activation_func,train_X_var ):
    model = Sequential()
    attention_flag=0
    model_input_LSTM=(train_X_var.shape[1],train_X_var.shape[2])
    for i in range(num_hidden_layers):
        if i==0: 
            #LSTM Expects following as input : [batch_size, lookback, feature]
            model.add(Bidirectional(LSTM(num_neurons_per_layer, return_sequences=True,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),
                                         activity_regularizer=L1L2(1e-5,1e-4),
                                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)), input_shape=model_input_LSTM))
            #model.add(LSTM(num_hidden_layers,activation=activation_func,return_sequences=True,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), input_shape=model_input_LSTM))
            model.add(Dropout(dropout_rate))
        else:
            #model.add(LSTM(num_neurons_per_layer,recurrent_dropout=dropout_rate,return_sequences=True,activation=activation_func))
            #model.add(Bidirectional(LSTM(num_neurons_per_layer, return_sequences=True,activation=activation_func)))
            model.add(Dense(num_hidden_layers,bias_regularizer=L1L2(l1=0.0, l2=0.0001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer='he_uniform'))
        
            #model.add(LSTM(num_hidden_layers,activation=activation_func,return_sequences=True,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), input_shape=(train_X_var.shape[1],train_X_var.shape[2])))
            #model.add(Attention1(return_sequences=True)) # receive 3D and output 3D
            if (attention_flag==0):
                model.add(attention(return_sequences=True)) # receive 3D and output 3D
                attention_flag+=1
            model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation='linear'))
    model.summary()
    return (model)


def get_LSTM_attention_model(num_hidden_layers, num_neurons_per_layer, dropout_rate, activation_func, train_X):
    with tf.device('/gpu:0'):
        # model_input = tf.keras.Input(shape=(train_X.shape[1],train_X.shape[2]))  # input layer.
        # for i in range(num_hidden_layers):
        #     #x = Bidirectional(LSTM(num_neurons_per_layer, return_sequences=True,bias_regularizer=L1L2(l1=0.0, l2=0.01),activity_regularizer=L1L2(1e-5,1e-4), activation=activation_func))(model_input)
        #     x = LSTM(num_neurons_per_layer, return_sequences=True,bias_regularizer=L1L2(l1=0.0, l2=0.01),activity_regularizer=L1L2(1e-5,1e-4), activation=activation_func)(model_input)            #x = Bidirectional(LSTM(num_neurons_per_layer, return_sequences=True,activation=activation_func))(model_input)
        #     x = Attention(num_neurons_per_layer/2)(x)
        #     x = layers.Dropout(dropout_rate)(x)
        # #x = Dense(1,activation='linear')(x)
        # outputs = layers.Dense(1, activation='linear')(x)
        # model = tf.keras.Model(inputs=model_input, outputs=outputs)
        # model.summary()
        model_input = tf.keras.Input(shape=(train_X.shape[1],train_X.shape[2]))  # input layer.
        for i in range(0,num_hidden_layers):
            if (i==0):
                #x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(model_input)
                #x = Bidirectional(LSTM(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)))(model_input)
                x = LSTM(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(model_input)
                x = layers.Dropout(dropout_rate)(x)
                #x = Attention(num_neurons_per_layer)(x)
                x = layers.Dropout(dropout_rate)(x)
            else:
                #x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(x)
                x = LSTM(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(x)
                x = layers.Dropout(dropout_rate)(x)
                #x = Attention(num_neurons_per_layer)(x)
                x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=model_input, outputs=outputs)
        model.summary()
    return model

def get_LSTM_attention_model_experiment(num_hidden_layers, num_neurons_per_layer, dropout_rate, activation_func, train_X):
    with tf.device('/gpu:0'):
        model_input = tf.keras.Input(shape=(train_X.shape[1],train_X.shape[2]))  # input layer.
        x = LSTM(num_neurons_per_layer, return_sequences=True,bias_regularizer=L1L2(l1=0.0, l2=0.01),activity_regularizer=L1L2(1e-5,1e-4), activation=activation_func)(model_input)            
        x = layers.Dropout(dropout_rate)(x)
        if (num_hidden_layers>1):
            for i in range(0,num_hidden_layers-1):
                x = LSTM(num_neurons_per_layer, return_sequences=True,bias_regularizer=L1L2(l1=0.0, l2=0.01),activity_regularizer=L1L2(1e-5,1e-4), activation=activation_func)(x)            
        x = Attention(num_neurons_per_layer/4)(x)
        # x = LSTM(num_neurons_per_layer,return_sequences=False, bias_regularizer=L1L2(l1=0.0, l2=0.01),activity_regularizer=L1L2(1e-5,1e-4), activation=activation_func)(x)            
        # x = Attention(num_neurons_per_layer/2)(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=model_input, outputs=outputs)
        model.summary()
    return model

def get_keras_LSTM_Model(num_hidden_layers, num_neurons_per_layer, dropout_rate, activation_func, train_X):
    with tf.device('/gpu:0'):
        model = Sequential()
        for i in range(num_hidden_layers):
            if i==0: 
                #LSTM Expects following as input : [batch_size, lookback, feature]
                #model.add(LSTM( num_neurons_per_layer,input_shape=(train_X.shape[1],train_X.shape[2]), return_sequences=True,activation=activation_func))   # 10 steps have been determined
                model.add(LSTM( num_neurons_per_layer,recurrent_dropout=dropout_rate, input_shape=(train_X.shape[1],train_X.shape[2]), return_sequences=True,activation=activation_func))
                model.add(Dropout(dropout_rate))
                model.add(LSTM(num_neurons_per_layer,recurrent_dropout=dropout_rate,return_sequences=True,activation=activation_func))
                model.add(Dropout(dropout_rate))
            else:
                model.add(Dense(1,activation='linear'))
                model.add(Dropout(dropout_rate))
        model.summary()
    return (model)




def get_ANN_attention_model_classification(num_hidden_layers, num_neurons_per_layer, dropout_rate, activation_func, train_X):
    with tf.device('/gpu:0'):
        model_input = tf.keras.Input(shape=(train_X.shape[1]))  # input layer.
        for i in range(num_hidden_layers):
            if(i==0):
                x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(model_input)
                x = layers.Dropout(dropout_rate)(x)
            else:
                x = layers.Dense(num_neurons_per_layer,activation=activation_func,bias_regularizer=L1L2(l1=0.0, l2=0.00001),activity_regularizer=L1L2(1e-5,1e-4),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42))(model_input)
                x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=model_input, outputs=outputs)
        model.summary()
    return (model)


#This function takes in the hyperparameters and returns a score (Cross validation).
def keras_mlp_cv_score(train_X,train_y,val_X,val_y,parameterization,weight=None):
    # model = get_keras_LSTM_Attention_Model(parameterization.get('num_hidden_layers'),
    #                         parameterization.get('neurons_per_layer'),
    #                         parameterization.get('dropout_rate'),
    #                         parameterization.get('activation'),train_X)
    model = get_ANN_attention_model_experiment(parameterization.get('num_hidden_layers'),
                            parameterization.get('neurons_per_layer'),
                            parameterization.get('dropout_rate'),
                            parameterization.get('activation'),train_X)
    # model = get_autoencoder_model_experiment(parameterization.get('num_hidden_layers'),
    #                         parameterization.get('neurons_per_layer'),
    #                         parameterization.get('dropout_rate'),
    #                         parameterization.get('activation'),train_X)
    opt = parameterization.get('optimizer')
    opt = opt.lower()
    learning_rate = parameterization.get('learning_rate')
    if opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt =='adamx':
        optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)     
    elif opt == 'rms':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    NUM_EPOCHS = 1000
    # Specify the training configuration.
    model.compile(optimizer=optimizer,loss=tf.keras.losses.MeanSquaredError(),metrics=['mse'])
    data = train_X
    labels = train_y# Response var values
    
    
    early_stop = EarlyStopping(monitor='val_loss', patience= 20, verbose=0, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=7, verbose=1, min_delta=1e-4, mode='min') #reduce learning rate when model is not improving
    callbacks_list=[early_stop,reduce_lr_loss] #Only early stop required for initial experiments
    
    # fit the model 
    res = model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=parameterization.get('batch_size'),validation_data=(val_X, val_y),callbacks=callbacks_list)    
    #res = model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=parameterization.get('batch_size'),validation_data=(val_X, val_y),callbacks=callbacks_list)    
    # look at the last 10 epochs. Get the mean and standard deviation of the validation score.
    last10_scores = np.array(res.history['val_loss'][-10:])
    mean = last10_scores.mean()
    sem = last10_scores.std()
    # If the model didn't converge then set a high loss.
    if np.isnan(mean):
        return 9999.0, 0.0
    return mean, sem

# Define the search space.
parameters=[    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [0.00001, 0.01],
        "log_scale": True,
    },
    {
        "name": "dropout_rate",
        "type": "range",
        "bounds": [0.001, 0.25],
        "log_scale": True,
    },
    {
        "name": "num_hidden_layers",
        "type": "range",
        "bounds": [1, 5],
        "value_type": "int"
    },
    {
        "name": "neurons_per_layer",
        "type": "range",
        "bounds": [400, 550],
        "value_type": "int"
    },
    {
        "name": "batch_size",
        "type": "choice",
        "values": [8,16,32,64,80],
    },    
    {
        "name": "activation",
        "type": "choice",
        "values": ['linear', 'relu'],
    },
    {
        "name": "optimizer",
        "type": "choice",
        "values": ['rms','adam','adamx', 'sgd'],
    },]

def evaluate(parameters,train_X,train_y,val_X, val_y):
    return {"keras_cv": keras_mlp_cv_score(train_X,train_y,val_X, val_y,parameters)}

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# In[1]:--------------------Outlier Detection-------------------------------------------------------------------
#Remove outliers greater than given standard deviation
from scipy import stats
def remove_outliers_std_dev(df,col_name, std_dev):
    df=df[col_name]
    out=df
    out=out[~((out-out.mean()).abs() > std_dev*out.std())] #Remove outliers that are more than given standard deviations away
    df=out
    return df    

def remove_outlier_IQR(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

def remove_outlier_Zscore(df, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, reduce=False) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)
    return df

def clip_outliers(df,minPercentile = 0.02,maxPercentile = 0.98):
    #df_reduced_corr= clip_outliers(df_reduced_corr,minPercentile = 0.02,maxPercentile = 0.98)
    df_list = list(df)
    numCols=len(df.columns)
    for _ in range(numCols):
        df[df_list[_]] = df[df_list[_]].clip((df[df_list[_]].quantile(minPercentile)),(df[df_list[_]].quantile(maxPercentile)))
    return df



def rolling_outlier_filter(data, mode='rolling', window=262, threshold=3):
    """Basic Filter.
function call : df_BTCdata_Close=rolling_outlier_filter(pd.Series(df_reduced_corr['df_BTCdata_Close']), 'rolling', 100,3)    
    
    Mark as outliers the points that are out of the interval:
    (mean - threshold * std, mean + threshold * std ).
    
    Parameters
    ----------
    data : pandas.Series
        The time series to filter.
    mode : str, optional, default: 'rolling'
        Whether to filter in rolling or expanding basis.
    window : int, optional, default: 262
        The number of periods to compute the mean and standard
        deviation.
    threshold : int, optional, default: 3
        The number of standard deviations above the mean.
        
    Returns
    -------
    series : pandas.DataFrame
        Original series and marked outliers.
    """
    msg = f"Type must be of pandas.Series but {type(data)} was passed."
    assert isinstance(data, pd.Series), msg
    
    series = data.copy()
    
    # rolling/expanding objects
    pd_object = getattr(series, mode)(window=window)
    mean = pd_object.mean()
    std = pd_object.std()
    
    upper_bound = mean + threshold * std
    lower_bound = mean - threshold * std
    
    outliers = ~series.between(lower_bound, upper_bound)
    # fill false positives with 0
    outliers.iloc[:window] = np.zeros(shape=window)
    
    series = series.to_frame()
    series['outliers'] = np.array(outliers.astype('int').values)
    series.columns = ['close', 'Outliers']
    
    return series


def make_classification_data(data,response_variable='close'):
    # Usage :  data=make_classification_data(data,'df_BTCdata_Close')
    # data = load_data()
    # Get price returns shifted by one period
    btc_returns=(data[response_variable]-data[response_variable].shift(1))/data[response_variable].shift(1)
    btc_returns = btc_returns.fillna(method='bfill')
    # Convert response variable to a categorical variable for classification
    category=[]
    for x in range(len(btc_returns)):
        if btc_returns[x]>=0:
            category.append(1)
        else:
            category.append(0)
        
    data.reset_index(drop=True, inplace=True)
    data['category']=pd.DataFrame(category)       
    data=data.drop('close',axis=1)
    data=data.rename(columns={'category': 'close'})
    data.isnull().sum()
    return data

# from math import factorial
# def savitzky_golay(y, window_size, order, deriv=0, rate=1):
#     try:
#         window_size = np.abs(np.int(window_size))
#         order = np.abs(np.int(order))
#     except ValueError:
#         raise ValueError("window_size and order have to be of type int")
#     if ((window_size % 2) != 1) or (window_size < 1):
#         raise TypeError("window_size size must be a positive odd number")
#     if window_size < order + 2:
#         raise TypeError("window_size is too small for the polynomials order")
#     order_range = range(order+1)
#     half_window = (window_size -1) // 2
#     # precompute coefficients
#     b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
#     m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
#     # pad the signal at the extremes with
#     # values taken from the signal itself
#     firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
#     lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
#     y = np.concatenate((firstvals, y, lastvals))
#     return np.convolve( m[::-1], y, mode='valid')

# def savitzky_golay_piecewise(xvals, data, kernel=11, order =4):
#     turnpoint=0
#     last=len(xvals)
#     if xvals[1]>xvals[0] : #x is increasing?
#         for i in range(1,last) : #yes
#             if xvals[i]<xvals[i-1] : #search where x starts to fall
#                 turnpoint=i
#                 break
#     else: #no, x is decreasing
#         for i in range(1,last) : #search where it starts to rise
#             if xvals[i]>xvals[i-1] :
#                 turnpoint=i
#                 break
#     if turnpoint==0 : #no change in direction of x
#         return savitzky_golay(data, kernel, order)
#     else:
#         #smooth the first piece
#         firstpart=savitzky_golay(data[0:turnpoint],kernel,order)
#         #recursively smooth the rest
#         rest=savitzky_golay_piecewise(xvals[turnpoint:], data[turnpoint:], kernel, order)
#         return numpy.concatenate((firstpart,rest))
    
    
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()
    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),initializer="zeros")
        super(attention,self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)
    # def call(self, x):
    #     e = K.tanh(K.dot(x,self.W)+self.b)
    #     a = K.softmax(e, axis=1)
    #     output = x*a
    #     if self.return_sequences:
    #         return output
    #     return K.sum(output, axis=1)
    
    
def partition_dataset(sequence_length, train_df):
    x, y = [], []
    nfut=3
    data_len = train_df.shape[0]
    for i in range(sequence_length, data_len - nfut + 1):
        x.append(train_df[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(train_df[i + nfut - 1, train_df['close']]) #contains the prediction values for validation (3rd column = Close), for single-step prediction
        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
    return x, y

def signaltonoise(a, axis=0, ddof=0):
    '''The signal-to-noise ratio of the input data.
Returns the signal-to-noise ratio of `a`, here defined as the mean
divided by the standard deviation.
Parameters'''
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
    