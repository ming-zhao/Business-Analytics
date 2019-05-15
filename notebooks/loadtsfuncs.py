import pandas as pd
from pandas.plotting import lag_plot
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pmdarima as pm
from ipywidgets import *

dataurl = 'https://raw.githubusercontent.com/ming-zhao/Business-Analytics/master/data/time_series/'

df_house = pd.read_csv(dataurl+'house_sales.csv', parse_dates=['date'], header=0, index_col='date')
df_house['year'] = [d.year for d in df_house.index]
df_house['month'] = [d.strftime('%b') for d in df_house.index]

df_drink = pd.read_csv(dataurl+'drink_sales.csv', parse_dates=['date'], header=0)
df_drink['date'] = [pd.to_datetime(''.join(df_drink.date.str.split('-')[i][-1::-1])) 
                       + pd.offsets.QuarterEnd(0) for i in df_drink.index]
df_drink = df_drink.set_index('date')
# df_drink[['q','year']]=df_drink['quarter'].str.split('-',expand=True)
df_drink['year'] = [d.year for d in df_drink.index]
df_drink['quarter'] = ['Q'+str(d.month//3) for d in df_drink.index]



noise = pd.Series(np.random.randn(200))
def randomwalk(drift):
    return pd.Series(np.cumsum(np.random.uniform(-1,1,(200,1)) + drift*np.ones((200,1))))

def plot_time_series(df, col_name, freq='Month', title=''):
    ax = df.plot(y=col_name, figsize=(15,6), x_compat=True)
    ax.set_xlim(pd.to_datetime(df.index[0]), 
                pd.to_datetime(str(pd.Timestamp(df.index[-1]).year+1) + '-01-01'))
    if freq=='Month':
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.title(title)
    plt.show()
    
def seasonal_plot(df, col_names, title=''):
    np.random.seed(100)
    years = pd.Series([x.year for x in df.index]).unique()
    mycolors = np.random.choice(list(mlp.colors.XKCD_COLORS.keys()), len(years), replace=False)
    
    plt.subplots(1, 1, figsize=(12,6), dpi=120)
    
    label_shift = .4
    if col_names[0]=='quarter':
        label_shift = .8
    
    for i, y in enumerate(years):
        if i > 0:        
            plt.plot(col_names[0], col_names[1], data=df.loc[df.year==y, :], color=mycolors[i], label=y)
            plt.text(df.loc[df.year==y, :].shape[0]-label_shift, 
                     df.loc[df.year==y, col_names[1]][-1:].values[0], y, color=mycolors[i], fontsize=12)
    plt.title(title)

def boxplot(df, col_names, title=''):
    fig, axes = plt.subplots(1, 2, figsize=(18,6), dpi=120)
    sns.boxplot(x='year', y=col_names[1], data=df, ax=axes[0])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30)
    sns.boxplot(x=col_names[0], y=col_names[1], data=df)

    axes[0].set_title('Year-wise Box Plot for {}\n(The Trend)'.format(title), fontsize=14); 
    axes[1].set_title('Month-wise Box Plot for {}\n(The Seasonality)'.format(title), fontsize=14)
    plt.show()

def analysis(df, y, x, printlvl):
    result = ols(formula=y+'~'+'+'.join(x), data=df).fit()
    if printlvl>=4:
        print(result.summary())
        print('\nstandard error of estimate:{:.5f}\n'.format(np.sqrt(result.scale)))
        
    if printlvl>=5:
        print("\nANOVA Table:\n", sm.stats.anova_lm(result, typ=2))    
    
    if printlvl>=1:
        if len(x)==1:
            fig, axes = plt.subplots(1,1,figsize=(8,5))
            sns.regplot(x=x[0], y=y, data=df,
                        ci=None, 
                        line_kws={'color':'green', 
                                  'label':"$Y$"+"$={:.2f}X+{:.2f}$\n$R^2$={:.3f}".format(result.params[1],
                                                                                         result.params[0],
                                                                                         result.rsquared)},
                        ax=axes);
            axes.legend()

    if printlvl>=2:
        fig, axes = plt.subplots(1,3,figsize=(20,6))
        axes[0].relim()
        sns.residplot(result.fittedvalues, result.resid , lowess=False, scatter_kws={"s": 80},
                      line_kws={'color':'r', 'lw':1}, ax=axes[0])
        axes[0].set_title('Residual plot')
        axes[0].set_xlabel('Fitted values')
        axes[0].set_ylabel('Residuals')
        axes[1].relim()
        stats.probplot(result.resid, dist='norm', plot=axes[1])
        axes[1].set_title('Normal Q-Q plot')
        axes[2].relim()
        sns.distplot(result.resid, ax=axes[2]);
        if printlvl==2:
            fig.delaxes(axes[1])
            fig.delaxes(axes[2])
    plt.show()    
    return result     
    
def stationarity_test(df_col, title=''):
    print('Test on {}:\n'.format(title))
    
    from statsmodels.tsa.stattools import adfuller, kpss
    # ADF Test
    result = adfuller(df_col.values, autolag='AIC')
    print('ADF Statistic \t: {:.5f}'.format(result[0]))
    print('p-value \t: {:.5f}'.format(result[1]))
    print('Critial Values \t:')
    for key, value in result[4].items():
        print('\t{:3.1f}% \t: {:.5f}'.format(float(key[:-1]), value))

    print('\nH0: The time series is non-stationary')
    if result[1]<0.05:
        print('We reject the null hypothesis at 5% level.')
    else:
        print('We do not reject the null hypothesis.')

    # KPSS Test
    result = kpss(df_col.values, regression='c')
    print('\nKPSS Statistic \t: {:.5f}'.format(result[0]))
    print('p-value \t: {:.5f}'.format(result[1]))
    print('Critial Values \t:')
    for key, value in result[3].items():
        print('\t{:3.1f}%\t: {:.5f}'.format(float(key[:-1]), value))

    print('\nH0: The time series is stationary')
    if result[1]<0.05:
        print('We reject the null hypothesis at 5% level.')
    else:
        print('We do not reject the null hypothesis.')
        
def decomp(df_col):
    from statsmodels.tsa.seasonal import seasonal_decompose
    import statsmodels.api as sm
    # Multiplicative Decomposition
    result_mul = sm.tsa.seasonal_decompose(df_col, model='multiplicative', extrapolate_trend='freq')
    print('Multiplicative Model\t: Observed {:.3f} = (Seasonal {:.3f} * Trend {:.3f} * Resid {:.3f})'.format(
        result_mul.observed[0], result_mul.trend[0], result_mul.seasonal[0], result_mul.resid[0]))

    # Additive Decomposition
    result_add = sm.tsa.seasonal_decompose(df_col, model='additive', extrapolate_trend='freq')
    print('Additive Model\t\t: Observed {:.3f} = (Seasonal {:.3f} + Trend {:.3f} + Resid {:.3f})'.format(
        result_mul.observed[0], result_add.trend[0], result_add.seasonal[0], result_add.resid[0]))

    # Setting extrapolate_trend='freq' takes care of any missing values
    #                                in the trend and residuals at the beginning of the series.
    plt.rcParams.update({'figure.figsize': (10,8)})
    result_mul.plot().suptitle('Multiplicative Decompose', fontsize=18)
    plt.subplots_adjust(top=.93)
    result_add.plot().suptitle('Additive Decompose', fontsize=18)
    plt.subplots_adjust(top=.93)
    plt.show()
    
def detrend(df_col, model = 'multiplicative'):
    # Using scipy: Subtract the line of best fit
    from scipy import signal
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose

    result_mul = sm.tsa.seasonal_decompose(df_col, model='multiplicative', extrapolate_trend='freq')
    result_add = sm.tsa.seasonal_decompose(df_col, model='additive', extrapolate_trend='freq')
    
    plt.subplots(1, 2, figsize=(12,4), dpi=80)
    detrended = signal.detrend(df_col.values)
    plt.subplot(1, 2, 1)
    plt.plot(detrended)
    plt.title('Subtracting the least squares fit', fontsize=16)

    if model=='multiplicative':
        detrended = df_col.values / result_mul.trend
    if model=='additive':
        detrended = df_col.values - result_add.trend
    plt.subplot(1, 2, 2)
    plt.plot(detrended)
    plt.title('Subtracting the trend component', fontsize=16)
    plt.show()
    
def deseasonalize(df_col, model, title=''):
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    plt.subplots(1, 1, figsize=(12,8))
    
    if model=='multiplicative' or model=='mul':
        result_mul = sm.tsa.seasonal_decompose(df_col, model='multiplicative', extrapolate_trend='freq')
        deseasonalized = df_col.values / result_mul.seasonal
    if model=='additive' or model=='add':
        result_add = sm.tsa.seasonal_decompose(df_col, model='additive', extrapolate_trend='freq')
        deseasonalized = df_col.values - result_add.seasonal
    
    plt.subplot(2,1,1)
    plt.plot(deseasonalized)
    plt.title('Deseasonalized {}'.format(title), fontsize=12)

def plot_autocorr(df_col, title=''):
    from pandas.plotting import autocorrelation_plot

    plt.rcParams.update({'figure.figsize':(8,3), 'figure.dpi':120})
    autocorrelation_plot(df_col.values)
    plt.title(title)
    plt.show()
    
def plot_acf_pacf(df_col, acf_lag, pacf_lag):
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1,2,figsize=(16,3), dpi=100)
    _ = plot_acf(df_col.values, lags=acf_lag, ax=axes[0])
    _ = plot_pacf(df_col.tolist(), lags=pacf_lag, ax=axes[1])
    
def differencing(df, col_name, title='', period=2):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    plt.rcParams.update({'figure.figsize':(9,6), 'figure.dpi':150})
    # Original Series
    fig, axes = plt.subplots(period+1, 2, sharex='col')
    fig.tight_layout()

    axes[0, 0].plot(df.index, df[col_name])
    axes[0, 0].set_title('Original Series')
    _ = plot_acf(df[col_name].values, lags=50, ax=axes[0, 1])
    print('Standard deviation original series: {:.3f}'.format(np.std(df['sales'].values)))
    
    for t in range(period):
        axes[t+1, 0].plot(df.index, df[col_name].diff(t+1))
        axes[t+1, 0].set_title('{}st Order Differencing'.format(t+1))
        plot_acf(df[col_name].diff(t+1).dropna(), lags=50, ax=axes[t+1, 1])
        print('Standard deviation {}st differencing: {:.3f}'.format(t+1,np.std(df['sales'].diff(t+1).dropna().values)))
        
    plt.title(title)
    plt.show()
    
def arima_(p, d, q):
    from statsmodels.tsa.arima_model import ARIMA

    model = ARIMA(df_house.sales, order=(p, d, q))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    # Plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1,2, figsize=(12,3))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()

    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
    model_fit.plot_predict(dynamic=False)
    plt.show()
    
def forecast_accuracy(forecast, actual):
    from statsmodels.tsa.stattools import acf
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(forecast - actual)[1]            # ACF1
    return({'MAPE':mape, 'ME':me, 'MAE': mae, 
            'MPE': mpe, 'RMSE':rmse, 'ACF1':acf1, 
            'Corr':corr, 'Minmax':minmax})

def arima_validation(p, d, q):
    from statsmodels.tsa.arima_model import ARIMA
    
    test_size = int(df_house.shape[0]*.25)

    train = df_house.sales[:-test_size]
    test = df_house.sales[-test_size:]
    model = ARIMA(train, order=(p, d, q))  
    model_fit = model.fit(disp=0)  
    print(model_fit.summary())
    
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1,2, figsize=(12,3))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()

    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
    
    # Forecast
    fc, se, conf = model_fit.forecast(test_size, alpha=0.05)  # 95% conf

    # Make as pandas series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    # Plot
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    
    print('{:7s}: {:8.4f}'.format('MAPE', forecast_accuracy(fc, test.values)['MAPE']))
    
def sarima_forcast(model, df, col_name, forecast_periods, freq):
    if freq=='month':
        periods = 12
    if freq=='quarter':
        periods = 4
    fitted, confint = model.predict(n_periods=forecast_periods, return_conf_int=True)
    if freq=='month':
        index_of_fc = pd.date_range(df.index[-1], periods = forecast_periods, freq='M')
    if freq=='quarter':
        index_of_fc = pd.date_range(df.index[-1], periods = forecast_periods, freq='3M')
        
    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.rcParams.update({'figure.figsize':(10,4), 'figure.dpi':120})
    plt.plot(df[col_name])
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index, 
                     lower_series, 
                     upper_series, 
                     color='k', alpha=.15)

    plt.title("SARIMAX Forecast of Drink Sales")
    plt.show()
    
def add_seasonal_index(df, col_name, freq='month', model='multiplicative'):
    from statsmodels.tsa.seasonal import seasonal_decompose
    if freq=='month':
        periods = 12
    if freq=='quarter':
        periods = 4
    seasonal_index = seasonal_decompose(df[col_name][-periods*3:],   # 3 years
                                    model=model, 
                                    extrapolate_trend='freq').seasonal[-periods:].to_frame()
    seasonal_index.columns = ['seasonal_index']
    if freq=='month':
        seasonal_index['month'] = [d.strftime('%b') for d in seasonal_index.index]
    if freq=='quarter':
        seasonal_index['quarter'] = ['Q'+str(d.month//3) for d in seasonal_index.index]
    df_tmp = pd.merge(df, seasonal_index, how='left', on=freq)
    df_tmp.index = df.index 
    return df_tmp

def sarimax_forcast(model, df, col_name, forecast_periods, freq):
    if freq=='month':
        periods = 12
    if freq=='quarter':
        periods = 4
    fitted, confint = model.predict(n_periods=forecast_periods, 
                                    exogenous=np.tile(df.seasonal_index[:periods],
                                                    forecast_periods//periods).reshape(-1,1),
                                    return_conf_int=True)
    if freq=='month':
        index_of_fc = pd.date_range(df.index[-1], periods = forecast_periods, freq='M')
    if freq=='quarter':
        index_of_fc = pd.date_range(df.index[-1], periods = forecast_periods, freq='3M')
        
    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.rcParams.update({'figure.figsize':(10,4), 'figure.dpi':120})
    plt.plot(df[col_name])
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index, 
                     lower_series, 
                     upper_series, 
                     color='k', alpha=.15)

    plt.title("SARIMAX Forecast of Drink Sales")
    plt.show()