import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from pandas import *
from scipy import interpolate

from statsmodels.tsa.seasonal import seasonal_decompose

pd.set_option('display.max_rows', 10000)


def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=24)
    rolstd = pd.rolling_std(timeseries, window=24)

    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries.Value.values.ravel(),1,autolag='AIC')
    #dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()



dateparse = lambda x: pd.datetime.strptime(x, "%d/%m/%Y %H:%M")
ts = pd.read_csv('data.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)
ts = ts[(ts.T != 0).any()] # drop zeros
ts = ts['2015':]
#ts_new_index = pd.date_range(ts.index.min(), ts.index.max(),freq='H');
#ts = ts.reindex(ts_new_index)
#ts = ts.astype(float)
#ts = ts.interpolate(method='linear')
ts_log = np.log(ts)

ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)


def runAll():
    #Remove outliers (or spikes) => no needed if using log
    #ts_log = ts[np.abs(ts.Value-ts.Value.mean())<=(1*ts.Value.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
    #ts_log = ts[~(np.abs(ts.Value-ts.Value.mean())>(3*ts.Value.std()))] #or if you prefer the other way around


    #moving_avg = pd.rolling_mean(ts_log,24)
    #ts_log_moving_avg_diff = ts_log - moving_avg
    #ts_log_moving_avg_diff.dropna(inplace=True)


    #Exponentially weighted moving average
    #expwighted_avg = pd.ewma(ts_log, halflife=24)
    #plt.plot(ts_log)
    #plt.plot(expwighted_avg, color='red')

    #Eliminating Trend and Seasonality



    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(ts_log,freq=24)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid


    ts_log_decompose = residual
    ts_log_decompose.dropna(inplace=True)
    #test_stationarity(ts_log_decompose)


    ###FORCASTING###
    from statsmodels.tsa.stattools import acf, pacf
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    #Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()

    from statsmodels.tsa.arima_model import ARIMA

    #AR Model
    #model = ARIMA(ts_log, order=(2, 1, 0))
    #results_AR = model.fit(disp=-1)
    #plt.plot(ts_log_diff)
    #plt.plot(results_AR.fittedvalues, color='red')
    #plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

    #MA Model
    #model = ARIMA(ts_log, order=(0, 1, 2))
    #results_MA = model.fit(disp=-1)
    #plt.plot(ts_log_diff)
    #plt.plot(results_MA.fittedvalues, color='red')
    #plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

    #Combined Model
    #model = ARIMA(ts_log, order=(2, 1, 2))
    #results_ARIMA = model.fit(disp=-1)
    #plt.plot(ts_log_diff)
    #plt.plot(results_ARIMA.fittedvalues, color='red')
    #plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

    ###Taking it back to original scale###
    #predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    #print predictions_ARIMA_diff.head()

    #predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    #print predictions_ARIMA_diff_cumsum.head()

    #predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
    #predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    #print(predictions_ARIMA_log.head())

    #predictions_ARIMA = np.exp(predictions_ARIMA_log)
    #plt.plot(ts)
    #plt.plot(predictions_ARIMA)
    #plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
    plt.show()


runAll()
