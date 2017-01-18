
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.relativedelta import *
from statsmodels.tsa.stattools import acf, pacf
from pandas import *

import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm

pd.set_option('display.max_rows', 10000)


def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=24).mean()
    rolstd = timeseries.rolling(window=24).std()

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
ts_original = pd.read_csv('data_old.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)
ts_original = ts_original['2016':]
ts_original = ts_original[(ts_original.T != 0).any()] # drop zeros
upsample = ts_original.resample('H').mean()
ts = upsample.interpolate(method='linear')
#print(ts)


#plt.plot(ts_original,color='blue')
#plt.plot(ts,color='red')
#plt.show()

#Remove outliers (or spikes) => no needed if using log
#ts_log = ts[np.abs(ts.Value-ts.Value.mean())<=(1*ts.Value.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
#ts_log = ts[~(np.abs(ts.Value-ts.Value.mean())>(3*ts.Value.std()))] #or if you prefer the other way around

#ts_log = np.log(ts)


def runAll():
    #moving_avg = pd.rolling_mean(ts_log,24)
    #ts_log_moving_avg_diff = ts_log - moving_avg
    #ts_log_moving_avg_diff.dropna(inplace=True)


    #Exponentially weighted moving average
    #expwighted_avg = pd.ewma(ts_log, halflife=24)
    #plt.plot(ts_log)
    #plt.plot(expwighted_avg, color='red')

    #Eliminating Trend and Seasonality
    decomposition = seasonal_decompose(ts)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    ts_log_decompose = residual
    ts_log_decompose.dropna(inplace=True)
    #test_stationarity(ts_log_decompose)

    ###FORCASTING EXPERIMENTAL###
    lag_acf = acf(ts_log_decompose, nlags=20)
    lag_pacf = pacf(ts_log_decompose, nlags=20, method='ols')



    ###FORCASTING###
    #lag_acf = acf(ts_log_diff, nlags=20)
    #lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    #Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_decompose)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_decompose)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    # #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_decompose)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_decompose)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
    #from statsmodels.tsa.arima_model import ARIMA

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
    #plt.show()

def runAll2():
    dateparse = lambda x: pd.datetime.strptime(x, "%d/%m/%Y %H:%M")
    ts_original = pd.read_csv('data_old.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)
    ts_original = ts_original['2016':]
    ts_original = ts_original[(ts_original.T != 0).any()] # drop zeros
    upsample = ts_original.resample('H').mean()
    ts = upsample.interpolate(method='linear')
    #df['first_difference'] = df.riders - df.riders.shift(1)

    ts_first_diff = ts - ts.shift(1)
    ts_first_diff.dropna(inplace=True)


    ts_seasonal_diff = ts - ts.shift(12)
    ts_seasonal_diff.dropna(inplace=True)

    seasonal_first_diff = ts_first_diff - ts_first_diff.shift(12)
    seasonal_first_diff.dropna(inplace=True)

    #test_stationarity(seasonal_first_diff)

    #PLOT
    #fig = plt.figure(figsize=(12,8))
    #ax1 = fig.add_subplot(211)
    #fig = sm.graphics.tsa.plot_acf(seasonal_first_diff.iloc[13:], lags=40, ax=ax1)
    #ax2 = fig.add_subplot(212)
    #fig = sm.graphics.tsa.plot_pacf(seasonal_first_diff.iloc[13:], lags=40, ax=ax2)
    #plt.show()

    #mod = sm.tsa.statespace.SARIMAX(ts, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
    #results = mod.fit()
    #print results.summary()

    #EXSISTING DATA FORCAST
    #ts_forcast = results.predict()
    #df[['riders', 'forecast']].plot(figsize=(12, 8))
    #plt.plot(ts['2017-01-01':],color='blue')
    #plt.plot(ts_forcast['2017-01-01':],color='red')
    #plt.show()
    #print(ts_forcast);

    #FORCAST
    start = datetime.strptime("2017-01-18", "%Y-%m-%d")
    date_list = [start + relativedelta(hours=x) for x in range(0,24)]
    future = pd.DataFrame(index=date_list, columns= ts.columns)
    df = pd.concat([ts, future])

    mod = sm.tsa.statespace.SARIMAX(df, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
    results = mod.fit()
    ts_forcast = results.predict()
    print(ts_forcast);
    plt.plot(ts['2017-01-16':],color='blue',label = 'Original')
    plt.plot(ts_forcast['2017-01-16':],color='red',label = 'Forcast')
    plt.legend(loc='best')
    plt.show()


runAll()

