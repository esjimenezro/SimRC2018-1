####### Download close prices data
def get_historical_closes(ticker, start_date, end_date=None):
    # Packages
    import pandas_datareader.data as web
    import pandas as pd
    import numpy as np
    #closes = web.YahooDailyReader(ticker, start_date, end_date).read().sort_index('major_axis')
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
def optimal_portfolio(daily_ret, n_opt, risk_free):
    # Frontier points
    #Packages
    import pandas as pd
    import sklearn.covariance as skcov
    import numpy as np
    import cvxopt as opt
    from cvxopt import blas, solvers
    num_stocks = daily_ret.columns.size   
    #cvxopt matrices
    robust_cov_matrix = skcov.ShrunkCovariance().fit(daily_ret).covariance_
    S = opt.matrix(robust_cov_matrix)
    daily_ret_mean = daily_ret.mean().values
    mus = np.linspace(daily_ret_mean.min(), daily_ret_mean.max(), n_opt)
    # Constraint matrices
    G = -opt.matrix(np.concatenate((np.array([daily_ret_mean]),np.eye(num_stocks)),axis=0))
    p = opt.matrix(np.zeros((num_stocks, 1)))
    A = opt.matrix(np.ones((1,num_stocks)))
    b = opt.matrix(np.array([1.0]))    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = np.zeros((n_opt, num_stocks))
    for k in range(n_opt):
        h = -opt.matrix(np.concatenate((np.array([[mus[k]]]),np.zeros((num_stocks,1))), axis=0))
        portfolios[k,:] = np.asarray(solvers.qp(S, p, G, h, A, b)['x']).T[0]
    # Risk and returns
    returns = 252*portfolios.dot(daily_ret_mean)
    risks = np.zeros(n_opt)
    for i in range(n_opt):
        risks[i] = np.sqrt(252*portfolios[i,:].dot(robust_cov_matrix).dot(portfolios[i,:].T))
    sharpe = (returns-risk_free)/risks
    return  pd.DataFrame(data=np.column_stack((returns,risks,sharpe,portfolios)),columns=(['Rendimiento','SD','Sharpe']+list(daily_ret.columns)))
####### Efficient frontier points via quadratic programming - with bond 
def optimal_portfolio_b(daily_ret, n_opt, risk_free, c0):
    # Frontier points
    #Packages
    import pandas as pd
    import sklearn.covariance as skcov
    import numpy as np
    import cvxopt as opt
    from cvxopt import blas, solvers
    # Bond inclusion
    robust_cov_matrix = np.insert((np.insert(skcov.ShrunkCovariance().fit(daily_ret).covariance_, daily_ret.columns.size, 0, axis=0)) , daily_ret.columns.size, 0, axis=1)
    daily_ret_b = pd.DataFrame(np.column_stack((np.asarray(daily_ret), c0*np.ones(daily_ret.index.size))), columns=list(daily_ret.columns)+['BOND'], index = daily_ret.index)
    num_stocks = daily_ret_b.columns.size
    daily_ret_mean = daily_ret_b.mean()
    mus = np.linspace(daily_ret_mean.min(), daily_ret_mean.max(), n_opt)
    #cvxopt matrices
    S = opt.matrix(robust_cov_matrix)
    G = -opt.matrix(np.concatenate((np.array([daily_ret_mean]),np.eye(num_stocks)),axis=0))
    p = opt.matrix(np.zeros((num_stocks, 1)))
    A = opt.matrix(np.ones((1,num_stocks)))
    b = opt.matrix(np.array([1.0])) 
    # Calculate efficient frontier weights using quadratic programming
    portfolios = np.zeros((n_opt, num_stocks))
    for k in range(n_opt):
        h = -opt.matrix(np.concatenate((np.array([[mus[k]]]),np.zeros((num_stocks,1))), axis=0))
        portfolios[k,:] = np.asarray(solvers.qp(S, p, G, h, A, b)['x']).T[0]
    # Risk and returns
    returns = 252*portfolios.dot(daily_ret_mean)
    risks = np.zeros(n_opt)
    for i in range(n_opt):
        risks[i] = np.sqrt(252*portfolios[i,:].dot(robust_cov_matrix).dot(portfolios[i,:].T))
    sharpe = (returns-risk_free)/risks
    return  pd.DataFrame(data=np.column_stack((returns,risks,sharpe,portfolios)),columns=(['Rendimiento','SD','Sharpe']+list(daily_ret_b.columns)))