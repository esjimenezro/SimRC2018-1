####### Download close prices data
def get_historical_closes(ticker, start_date, end_date=None):
    # Packages
    import pandas_datareader.data as web
    import pandas as pd
    import numpy as np
    #closes = web.DataReader(name=ticker, data_source='yahoo', start=start_date, end=end_date).sort_index('major_axis')
    closes = web.YahooDailyReader(symbols=ticker, start=start_date, end=end_date).read()
    closes.set_axis(closes.loc['date',:,ticker[0]].values, axis=1, inplace=True)
    closes = closes.loc['adjclose'].sort_index().dropna()
    closes = pd.DataFrame(np.array(closes.as_matrix(), dtype=np.float64), columns=ticker, index=closes.index)
    closes.index.name = 'Date'
    #return closes.loc['Adj Close']
    return closes
####### Calculation of log-returns
def calc_daily_ret(closes):
    import numpy as np
    return np.log(closes/closes.shift(1)).dropna()
####### Montecarlo simulation of portfolios to obtain the efficient frontier
def sim_mont_portfolio(daily_ret, num_portfolios, risk_free):
    num_stocks = daily_ret.columns.size
    #Packages
    import pandas as pd
    import sklearn.covariance as skcov
    import numpy as np
    # Mean returns
    daily_ret_mean = daily_ret.mean()
    # Covariance matrix
    robust_cov_matrix = skcov.ShrunkCovariance().fit(daily_ret).covariance_
    #Simulated weights
    weights = np.random.random((num_portfolios, num_stocks))
    weights /= np.sum(weights, axis=1)[:, None]
    portfolio_ret = weights.dot(daily_ret_mean) * 252
    portfolio_std_dev = np.zeros(num_portfolios)
    for i in range(num_portfolios):
        portfolio_std_dev[i]=np.sqrt(252*(((weights[i,:]).dot(robust_cov_matrix)).dot(weights[i,:].T))) 
    sharpe = (portfolio_ret-risk_free)/portfolio_std_dev
    return pd.DataFrame(np.column_stack((portfolio_ret,portfolio_std_dev,sharpe,weights)),columns=(['Rendimiento','SD','Sharpe']+list(daily_ret.columns)))
####### Efficient frontier points via quadratic programming
#def optimal_portfolio(daily_ret, n_opt, risk_free):
#    # Frontier points
#    #Packages
#    import pandas as pd
#    import sklearn.covariance as skcov
#    import numpy as np
#    import cvxopt as opt
#    from cvxopt import blas, solvers
#    num_stocks = daily_ret.columns.size
#    mus = [(10**(5.0 * t/N- 1.0)-10**(-1)) for t in range(N)]   
#    #cvxopt matrices
#    S = opt.matrix(skcov.ShrunkCovariance().fit(daily_ret).covariance_)
#    daily_ret_mean = daily_ret.mean().values
#    # Constraint matrices
#    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
#    h = opt.matrix(0.0, (n ,1))
#    A = opt.matrix(np.array(np.ones(num_stocks),daily_ret_mean), (2, num_stocks))
#    b = opt.matrix(np.array(1.0))    
#    # Calculate efficient frontier weights using quadratic programming
#    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
#    # Risk and returns
#    returns = [252*blas.dot(pbar, x) for x in portfolios]
#    risks = [np.sqrt(252*blas.dot(x, S*x)) for x in portfolios]
#    portfolios=[np.eye(n).dot(portfolios[i])[:,0] for i in range(N)]
#    returns = np.asarray(returns)
#    risks = np.asarray(risks)
#    sharpe=np.divide((returns-r),risks) 
#    portfolios = np.asarray(portfolios)
#    return  pd.DataFrame(data=np.column_stack((returns,risks,sharpe,portfolios)),columns=(['Returns','SD','Sharpe']+list(daily_returns.columns)))