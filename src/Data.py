##
# 用来统一地进行数据处理
# @author zzn
##

import pandas as pd
from .utils import utils
from .CPD import CPD


class DataProcessor:
    ''' 获取需要训练 LSTM 网络的输入数据和目标拟合数据
    
    Attributes:
        stock: utils类中绝大多数的方法面向的股票，形式为特定股票的代码
        start_date: 考察股票行为和数据时指定的起始日期，用以支持utils类中部分进行区间操作和查询的方法
        end_date: 考察股票行为和数据时指定的截止日期，用以支持utils类中部分进行区间操作和查询的方法
    '''
    ##
    # protected
    # 以下方法均为类内保护方法，使用该类的用户不应该在类以外的任何地方调用这些方法
    ##
    def _mergeData(self, *single_dataset) -> list:
        ''' 输入要合并的单个的数据集，该方法把每个单独的数据集各自对应的项归并成一个元组，然后加入新的数据集中
    
        Arg:
            单独的数据集

        Returns:
            归并后的数据集
        '''
        merged_data = []
        for ele in zip(*single_dataset):
            merged_data.append(ele)
        return merged_data
    ##
    # public
    # 以下方法为公有方法，使用该类的用户仅可以在类外调用该类的公有方法
    ##
    def __init__(self, stock_code: str, start_date: str, end_date: str, interval: int):
        ''' 需要用户制定获取数据的起止日期以及面向的股票
        
        Arg:
            stock: 股票代码
            start_date: 起始日期，格式为“年-月-日”，如"2022-05-16"
            end_date: 截止日期，格式为“年-月-日”，如"2022-05-16"
            interval: 指定后续绝大多数操作的时间窗口
        '''
        # 检验日期格式是否合法
        import re
        date_regex = '[1-2][0-9]{3}-(0[0-9]|1[0-2])-([0-2][0-9]|3[0-1])'
        if re.match(date_regex, start_date) is None or re.match(date_regex, end_date) is None:
            raise ValueError('请将日期格式更正为“年-月-日”，如"1970-01-01"，并检查输入日期是否合法！')
        # 检验日期内容是否合法
        try:
            self.__start_date = pd.Timestamp(start_date)
            self.__end_date = pd.Timestamp(end_date)
        except ValueError:
            raise ValueError('请输入正确格式与内容的日期！\n')
        # 初始化
        self.__stock_code = stock_code
        self.__base = utils(self.__stock_code, self.__start_date, self.__end_date)
        self.__interval = interval
    
    def getTrainData(self) -> list:
        ''' 获取 LSTM 网络所需的训练数据集，但在真实使用前还需要根据输入规模进行reshape工作
        
        Returns:
            LSTM_Model类的fit方法用到的训练集
        '''
        CPD_info = (
            CPD().getCPDScoreAndLocation(self.__stock_code, self.__start_date, self.__end_date, self.__interval))
        # 训练所需的单独数据集
        normalized_returns = self.__base.getNormalizedReturns(self.__interval)
        macd_indicators = self.__base.getMACD()
        CPD_scores = [window[0][0] for window in CPD_info for i in range(len(window))]
        CPD_locations = [window[i][1] for window in CPD_info for i in range(len(window))]
        # 数据归并
        train_data = self._mergeData(normalized_returns, macd_indicators, CPD_scores, CPD_locations)
        return train_data
    
    def getTargetData(self) -> list:
        ''' 获取 LSTM 网络所需的目标数据集
        
        Returns:
            LSTM_Model类的fit方法用到的目标集
        '''
        # 拟合所需单独的目标数据集
        volatilities = self.__base.getVolRange(60, 0.94)
        oneday_lagged_returns = self.__base.getOneDayLaggedReturns()
        # 归并
        target_data = self._mergeData(volatilities, oneday_lagged_returns)
        return target_data
