##
# 基础组建库，提供对 mindgo-api 的简单封装
# @author zzn
##

import warnings
import pandas as pd
from datetime import timedelta
from modules.mindgo_api import *


class utils:
    ''' 一些基础组件，对mindgo提供的部分股票api进行了封装，根据需求简化了一些常用数据请求方法的接口，实现了一些简单的数据变换函数
    
    Attributes:
          stock: utils类中绝大多数的方法面向的股票，形式为特定股票的代码
          start_date: 考察股票行为和数据时指定的起始日期，用以支持utils类中部分进行区间操作和查询的方法
          end_date: 考察股票行为和数据时指定的截止日期，用以支持utils类中部分进行区间操作和查询的方法
    '''
    ##
    # protected
    # 以下方法均为类内保护方法，使用该类的用户不应该在类以外的任何地方调用这些方法
    ##
    def _getPrice(self, stock_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
        ''' 获取给定股票在给定日期区间上的收盘价
            API灵活，不受类的初始化参数限制
        
        params: stock: 股票代码
                start_date: 起始日期
                end_date: 截止日期
                
        return: 给定股票在给定日期区间上的收盘价的序列
        '''
        # 需要校验给定日期是否为交易日
        price_df = get_price([stock_code],
                             start_date=start_date,
                             end_date=end_date,
                             fre_step='1d', fields=['close'])
        price = price_df[stock_code]['close']
        # 通过判断 start_date -> end_date 的日期是否满足交易日的逻辑相对复杂
        # 根据 mindgo get_price api的特性，非交易日数据默认为空，所以根据返回值 price 的特性判断日期是否满足交易日
        if price.size == 0:
            warnings.warn('用户提供的日期区间均为非交易日或给定的股票代码无相应数据，价格将返回为空序列!', UserWarning, stacklevel=1)
        return price
    
    def _getOneDayLaggedReturns(self, stock_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
        ''' 给定特定的股票以及考察时间段，获得该时间段内当期交易日所得回报率r_t滞后一天的回报率r_t+1
            API灵活，不受类的初始化参数限制
        
        params: stock: 请求的回报率对应的股票
                start_date: 当期起始时间点
                end_date: 当期结束时间点
        
        return: 给定股票在当期滞后一天的回报率序列
        '''
        # 开始日期根据date真实能够请求到的交易数据的当日来计算最近滞后的交易日，本质上是先校准date，再相应校准date+1
        start_date = self.getNearestTradeDayLagged(start_date)
        # 结束日期只需要校准到最近的交易日即可，即校准date一趟便可
        end_date += timedelta(1)
        while not self.isTradeDay(end_date):
            end_date += timedelta(1)
        # 请求滞后期的回报率数据
        returns_lagged = self._getPrice(stock_code, start_date, end_date)
        return returns_lagged
    ##
    # public
    # 以下方法为公有方法，使用该类的用户仅可以在类外调用该类的公有方法
    ##
    def __init__(self, stock_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
        ''' 初始化 utils 类，需要提供股票代码，考察的起始日期以及终止日期，后续相关操作将针对用户给定的股票并建立在提供的时间区间上
        
        params: stock: 股票代码
                start_date: 起始日期
                end_date: 截止日期
        '''
        # 输入数据类型校验
        if (not isinstance(stock_code, str) or
            not isinstance(start_date, pd.Timestamp) or
            not isinstance(end_date, pd.Timestamp)):
            raise TypeError('请检查utils类初始化时给定的参数类型是否合法!')
        # 输入合法性校验
        if get_security_info(stock_code) is None:
            raise ValueError('股票代码无效!')
        if start_date > end_date:
            raise ValueError('起始日期需不晚于截止日期!')
        # 初始化
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date

    def getPrice(self) -> pd.Series:
        ''' 给定股票代码，起始日期与截止日期，返回起始日期到截止日期之间的交易日中该股票的收盘价

        return: price: 指定日期期间交易日该股票的每日收盘价
        '''
        return self._getPrice(self.stock_code, self.start_date, self.end_date)

    def getClosePriceByDate(self, date: pd.Timestamp) -> float:
        ''' 对 get_price API 的一层包装，获取某支股票指定日期的数据，若指定日期无数据，则顺延一天
        
        params: date: 需要请求当日收盘价的日期
        
        return: close_price: 指定股票在指定日期的收盘价
        '''
        try:
            close_price_df = get_price([self.stock_code],
                                    start_date=date,
                                    end_date=date, 
                                    fre_step='1d', fields=['close'])
            close_price = close_price_df[self.stock_code]['close'][0]
            return close_price
        except IndexError:
            return self.getClosePriceByDate(date + timedelta(1))
        
    def getMACD(self) -> pd.Series:
        ''' 给定股票代码，起始日期与截止日期，返回起始日期到截止日期之间的交易日中该股票的MACD因子值
        
        return: macd: 指定股票在指定日期的MACD因子值
        '''
        trade_days = get_trade_days(self.start_date, self.end_date)
        # 请求数据
        q = query(
                factor.symbol, factor.macd
            ).filter(
                factor.symbol == self.stock_code,
                factor.date.in_(trade_days.strftime('%Y-%m-%d'))
                )
        macd = get_factors(q)['factor_macd']
        # 日期区间均为非交易日时给出警告
        if macd.size == 0:
            warnings.warn('用户提供的日期区间均为非交易日或给定的股票代码无相应数据，MACD因子值将返回为空序列!', UserWarning, stacklevel=1)
        return macd
    
    def isTradeDay(self, date: pd.Timestamp) -> bool:
        ''' 判断给定日期是否为交易日
        
        params: date: 需被判断是否为交易日的日期
        
        return: 为真，若date是交易日
        '''
        return len(get_trade_days(start_date=date, end_date=date)) == 1
    
    def getNearestTradeDayLagged(self, date: pd.Timestamp) -> pd.Timestamp:
        ''' 给定时间t，在t请求到的收盘价基础上请求下一天的收盘价
            若t为交易日，t+1日为交易日，此时t请求的是恰好是时刻t的收盘价，则该函数返回t+1日该股票的收盘价
            若t为交易日，t+1日为非交易日，此时t请求的是恰好是时刻t的收盘价，则该函数返回t后最近一期交易日的收盘价
            若t为非交易日，t+1日为交易日，此时t请求的是恰好是时刻t+1的收盘价，则该函数在t+1日的基础上求顺延一日的收盘价
            若t为非交易日，t+1日也为非交易日，此时t请求的是t后最近一期交易日的收盘价，则该函数返回最近一期基础上顺延得到的收盘价
        
        params: date: 给定的日期t
        
        return: 最近滞后一期的交易日的收盘价
        '''
        # 如果基期是交易日
        if self.isTradeDay(date):
            # 滞后一天
            date += timedelta(1)
            # 需判断滞后日是否为交易日，若非交易日，则需要将日期调整为滞后的最近交易日
            while not self.isTradeDay(date):
                date += timedelta(1)
            return date
        # 如果基期是休息日
        # 根据下一天的情况判断，如果下一天是工作日，则按照上一个if中对工作日的处理方法处理，若下一天仍是休息日，则继续递归
        else:
            return self.getNearestTradeDayLagged(date + timedelta(1))
    
    def getOneDayLaggedReturns(self) -> pd.Series:
        ''' 根据类的初始化参数，求给定股票在给定时间段上整体滞后一天的回报率序列
        
        return: 定股票在给定时间段上整体滞后一天的回报率序列
        '''
        return self._getOneDayLaggedReturns(self.stock_code, self.start_date, self.end_date)
        
    def getVolatility(self, date: pd.Timestamp, interval: int, alpha: float) -> float:
        ''' 计算给定股票的在给定日期波动率，计算方法为EWMA
        
        params: date: 给定的日期
                interval: EWMA方法回望的时间期限
                alpha: EWMA中计算波动率的配权
                
        return: 给定股票的在给定日期按EWMA方法计算得到的波动率
        '''
        from numpy import log, square, mean, sqrt
        # 查询距离基期最近的交易日，向前搜索
        begining_date = date - timedelta(interval)
        while not self.isTradeDay(begining_date):
            begining_date -= timedelta(1)
        # 请求收盘价
        return_price = self._getPrice(self.stock_code, begining_date, date)
        size = return_price.size
        log_return = [log(return_price[i + 1] / return_price[i]) for i in range(size - 1)]
        # 使用均方作为初始值
        sigma_square_0 = mean(square(log_return)) 
        # 定义递归计算ewma下的波动率
        def calc_sigma_square_recursive(t, λ, r):
            if t == 0:
                return sigma_square_0
            calc = calc_sigma_square_recursive
            return (1 - λ) * square(log_return[-1]) + λ * calc(t - 1, λ, r[: -1])
        # 计算波动率
        volatility = sqrt(calc_sigma_square_recursive(size, 0.94, log_return))
        return volatility
    
    def getVolRange(self, interval: int, alpha: float) -> pd.Series:
        ''' 获得给定股票在给定日期区间上的波动率，根据EWMA方法计算得到
        
        params: interval: EWMA方法回望的时间期限
                alpha: EWMA中计算波动率的配权
        
        return: 给定股票在给定日期区间的波动率
        '''
        date = self.start_date
        vols = []
        while date <= self.end_date:
            if not self.isTradeDay(date):
                date += timedelta(1)
                continue
            vols.append(self.getVolatility(date, interval, alpha))
            date += timedelta(1)
        return pd.Series(vols)
    
    def standardize(self, vector: pd.Series) -> pd.Series:
        ''' 对给定的向量进行标准化
        
        params: vector: 需要被标准化的向量, 一维
        
        return: 标准化变换后的向量
        '''
        # 判断类型并转换
        if not type(vector) is pd.Series:
            warnings.warn('给定将被标准化的参数vector非"pd.Series"类型，即将发生自动类型转换！', UserWarning, stacklevel=1)
            vector = pd.Series(vector)
        std_vector = []
        E = np.mean(vector)
        var = np.var(vector)
        for ele in vector:
            std_vector.append((ele - E) / np.sqrt(var))
        std_vector = pd.Series(std_vector)
        std_vector.index = vector.index
        return std_vector
    
    def getNormalizedReturns(self, interval: int) -> pd.Series:
        ''' 对 uitls 初始化时给定的股票以及时间段上的回报率序列进行按照 interval 设置的回望期进行正规化
        
        params: returns: 原始的回报率序列，要求 Series的index属性返回的序列中元素类型为 pd.Timestamp
        
        return: 正规化后的回报率序列
        '''
        # 初始化
        from numpy import sqrt
        returns = self.getPrice()
        normalized_returns = []
        # 逐一正规化
        for date in returns.index:
            current_price = returns[date]
            look_back_date = date - timedelta(interval)
            # 校准交易日
            while not self.isTradeDay(look_back_date):
                look_back_date -= timedelta(1)
            # 请求价格数据并计算回顾期的回报率
            look_back_price = self._getPrice(self.stock_code, look_back_date, look_back_date).tolist()[0]
            look_back_return = (current_price - look_back_price) / look_back_price
            # 请求当前波动率
            vol = self.getVolatility(date, interval, 0.94)
            # 计算正规化后的回报率并归入正规化回报率序列      
            normalized_returns.append(look_back_return / vol * sqrt(interval))
        return pd.Series(normalized_returns)