##
# 对 packages 下的 utils.py 提供简单的单元测试以及集成测试
# @author zzn
##

import unittest
from packages.Test import Test
from src.Utils import *

##
# 单元测试
##
@Test
class testGetPrice(unittest.TestCase):
    def test_one_trade_day(self):
        date = pd.Timestamp('2022-03-15')    
        func = utils(self.stock_code, date, date)
        
        expected = [2.93]
        actual = func.getPrice().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_one_nontrade_day(self):
        date = pd.Timestamp('2022-04-03')    
        func = utils(self.stock_code, date, date)
        
        expected = []
        actual = func.getPrice().tolist()
        
        self.assertEqual(expected, actual)
        
    def test_nontrade_days(self):
        start = pd.Timestamp('2022-04-02') 
        end = pd.Timestamp('2022-04-05')
        func = utils(self.stock_code, start, end)
        
        expected = []
        actual = func.getPrice().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_leftside_non_trade_day(self):
        start = pd.Timestamp('2022-04-02') 
        end = pd.Timestamp('2022-04-08')
        func = utils(self.stock_code, start, end)
        
        expected = [3.10, 3.09, 3.10]
        actual = func.getPrice().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_rightside_non_trade_day(self):
        start = pd.Timestamp('2022-03-30') 
        end = pd.Timestamp('2022-04-04')
        func = utils(self.stock_code, start, end)
        
        expected = [3.05, 3.08, 3.08]
        actual = func.getPrice().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_twosides_non_trade_day_with_inner_trade_days(self):
        start = pd.Timestamp('2022-04-03') 
        end = pd.Timestamp('2022-04-10')
        func = utils(self.stock_code, start, end)
        
        expected = [3.10, 3.09, 3.10]
        actual = func.getPrice().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_ordered_trade_days(self):
        start = pd.Timestamp('2022-03-10') 
        end = pd.Timestamp('2022-03-17')
        func = utils(self.stock_code, start, end)
        
        expected = [2.94, 2.97, 2.97, 2.93, 2.94, 2.93]
        actual = func.getPrice().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_unordered_trade_days(self):
        start = pd.Timestamp('2022-03-17')
        end = pd.Timestamp('2022-03-10') 
        
        try:
            func = utils(self.stock_code, start, end)
            raise AssertionError('未能正常抛出异常')
        except Exception as e:
            self.assertEqual(str(e), '起始日期需不晚于截止日期!')
    
    def runTest(self):
        self.stock_code = '601288.SH' # 农业银行
        self.test_one_trade_day()
        self.test_one_nontrade_day()
        self.test_nontrade_days()
        self.test_leftside_non_trade_day()
        self.test_rightside_non_trade_day()
        self.test_twosides_non_trade_day_with_inner_trade_days()
        self.test_ordered_trade_days()
        self.test_unordered_trade_days()

@Test
class testGetClosePriceByDate(unittest.TestCase):
    def test_trade_day(self):
        date = pd.Timestamp('2022-04-01')
        
        func = utils(self.stock_code, date, date)
        
        expected = 3.08 # 应该是 2022-04-01 的收盘价
        actual = func.getClosePriceByDate(date).tolist()
        
        self.assertEqual(expected, actual)
    
    def test_nontrade_day(self):
        date = pd.Timestamp('2022-04-02')
        
        func = utils(self.stock_code, date, date)
        
        expected = 3.10 # 应该是 2022-04-06 的收盘价
        actual = func.getClosePriceByDate(date).tolist()
        
        self.assertEqual(expected, actual)
    
    def runTest(self):
        self.stock_code = '601288.SH' # 农业银行
        self.test_trade_day()
        self.test_nontrade_day()

@Test
class testGetMACD(unittest.TestCase):
    def test_one_trade_day(self):
        date = pd.Timestamp('2022-03-15')    
        func = utils(self.stock_code, date, date)
        
        expected = [-0.0137]
        actual = func.getMACD().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_one_nontrade_day(self):
        date = pd.Timestamp('2022-04-02')    
        func = utils(self.stock_code, date, date)
        
        expected = []
        actual = func.getMACD().tolist()
        
        self.assertEqual(expected, actual)
        
    def test_nontrade_days(self):
        start = pd.Timestamp('2022-04-02') 
        end = pd.Timestamp('2022-04-05')
        func = utils(self.stock_code, start, end)
        
        expected = []
        actual = func.getMACD().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_leftside_non_trade_day(self):
        start = pd.Timestamp('2022-04-02') 
        end = pd.Timestamp('2022-04-08')
        func = utils(self.stock_code, start, end)
        
        expected = [0.0311, 0.0294, 0.0279]
        actual = func.getMACD().tolist()
        
        self.assertEqual(expected, actual)
        
    def test_rightside_non_trade_day(self):
        start = pd.Timestamp('2022-03-30') 
        end = pd.Timestamp('2022-04-04')
        func = utils(self.stock_code, start, end)
        
        expected = [0.0217, 0.0271, 0.0290]
        actual = func.getMACD().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_twosides_non_trade_day_with_inner_trade_days(self):
        start = pd.Timestamp('2022-04-03') 
        end = pd.Timestamp('2022-04-10')
        func = utils(self.stock_code, start, end)
        
        expected = [0.0311, 0.0294, 0.0279]
        actual = func.getMACD().tolist()
        
        self.assertEqual(expected, actual) 
    
    def test_ordered_trade_days(self):
        start = pd.Timestamp('2022-03-10') 
        end = pd.Timestamp('2022-03-17')
        func = utils(self.stock_code, start, end)
        
        expected = [-0.0184, -0.0145, -0.0113, -0.0137, -0.0131, -0.0131]
        actual = func.getMACD().tolist()
        
        self.assertEqual(expected, actual)
    
    def runTest(self):
        self.stock_code = '601288.SH' # 农业银行
        self.test_one_trade_day()
        self.test_one_nontrade_day()
        self.test_nontrade_days()
        self.test_leftside_non_trade_day()
        self.test_rightside_non_trade_day()
        self.test_twosides_non_trade_day_with_inner_trade_days()
        self.test_ordered_trade_days()

@Test
class testIsTradeDay(unittest.TestCase):
    def test_trade_day(self):
        date = pd.Timestamp('2022-02-16')
        func = utils(self.stock_code, date, date)
        self.assertTrue(func.isTradeDay(date))
        
    def test_nontrade_day(self):
        date = pd.Timestamp('2022-02-26')
        func = utils(self.stock_code, date, date)
        self.assertFalse(func.isTradeDay(date))
    
    def runTest(self):
        self.stock_code = '601288.SH' # 农业银行
        self.test_trade_day()
        self.test_nontrade_day()

@Test
class testGetNearestTradeDayLagged(unittest.TestCase):
    def test_t_is_trade_day1(self):
        # t交易日，t+1交易日
        date = pd.Timestamp('2022-03-28')
        func = utils(self.stock_code, date, date)
        
        expected = pd.Timestamp('2022-03-29')
        actual = func.getNearestTradeDayLagged(date)
        
        self.assertEqual(expected, actual)
    
    def test_t_is_trade_day2(self):
        # t交易日，t+1非交易日，顺延到最近一期
        date = pd.Timestamp('2022-04-29')
        func = utils(self.stock_code, date, date)
        
        expected = pd.Timestamp('2022-05-05')
        actual = func.getNearestTradeDayLagged(date)
        
        self.assertEqual(expected, actual)
        
    def test_t_is_nontrade_day1(self):
        # t非交易日，t+1交易日，t+1顺延一期
        date = pd.Timestamp('2022-03-13')
        func = utils(self.stock_code, date, date)
        
        expected = pd.Timestamp('2022-03-15')
        actual = func.getNearestTradeDayLagged(date)
        
        self.assertEqual(expected, actual)
        
    def test_t_is_nontrade_day2(self):
        # t非交易日，t+1非交易日，顺延的顺延
        date = pd.Timestamp('2022-02-26')
        func = utils(self.stock_code, date, date)
        
        expected = pd.Timestamp('2022-03-01')
        actual = func.getNearestTradeDayLagged(date)
        
        self.assertEqual(expected, actual)
    
    def runTest(self):
        self.stock_code = '601288.SH' # 农业银行
        self.test_t_is_trade_day1()
        self.test_t_is_trade_day2()
        self.test_t_is_nontrade_day1()
        self.test_t_is_nontrade_day2()

@Test
class testGetOneDayLaggedReturns(unittest.TestCase):
    def test_one_trade_day(self):
        date = pd.Timestamp('2022-03-15')    
        func = utils(self.stock_code, date, date)
        
        expected = [503.10]
        actual = func.getOneDayLaggedReturns().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_one_nontrade_day(self):
        date = pd.Timestamp('2022-04-03')
        func = utils(self.stock_code, date, date)
        
        expected = []
        actual = func.getOneDayLaggedReturns().tolist()
        
        self.assertEqual(expected, actual)
        
    def test_nontrade_days(self):
        start = pd.Timestamp('2022-04-02') 
        end = pd.Timestamp('2022-04-05')
        func = utils(self.stock_code, start, end)
        
        expected = []
        actual = func.getOneDayLaggedReturns().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_leftside_non_trade_day(self):
        start = pd.Timestamp('2022-04-02') 
        end = pd.Timestamp('2022-04-08')
        func = utils(self.stock_code, start, end)
        
        expected = [495.22, 495.00, 459.00]
        actual = func.getOneDayLaggedReturns().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_rightside_non_trade_day(self):
        start = pd.Timestamp('2022-03-30') 
        end = pd.Timestamp('2022-04-04')
        func = utils(self.stock_code, start, end)
        
        expected = [512.30, 518.90, 508.17]
        actual = func.getOneDayLaggedReturns().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_twosides_non_trade_day_with_inner_trade_days(self):
        start = pd.Timestamp('2022-04-03') 
        end = pd.Timestamp('2022-04-10')
        func = utils(self.stock_code, start, end)
        
        expected = [495.22, 495.00, 459.00]
        actual = func.getOneDayLaggedReturns().tolist()
        
        self.assertEqual(expected, actual)
    
    def test_ordered_trade_days(self):
        start = pd.Timestamp('2022-03-10') 
        end = pd.Timestamp('2022-03-17')
        func = utils(self.stock_code, start, end)
        
        expected = [493.55, 463.00, 463.20, 503.10, 524.50, 510.50]
        actual = func.getOneDayLaggedReturns().tolist()
        
        self.assertEqual(expected, actual)
    
    
    def runTest(self):
        self.stock_code = '300750.SZ'
        self.test_one_trade_day()
        self.test_one_nontrade_day()
        self.test_nontrade_days()
        self.test_leftside_non_trade_day()
        self.test_rightside_non_trade_day()
        self.test_twosides_non_trade_day_with_inner_trade_days()
        self.test_ordered_trade_days()

##
# 集成测试
##
@Test
class testGetNormalizedReturns(unittest.TestCase):
    def testNoneReturnSeries(self):
        start = pd.Timestamp('2022-04-03') 
        end = pd.Timestamp('2022-04-03')
        func = utils(self.stock_code, start, end)
        
        expected = pd.Series([])
        actual = func.getNormalizedReturns(self.interval)
        
        self.assertEqual(expected.all(), actual.all())
        
    def testNormalSituation(self):
        import matplotlib.pyplot as plt
        start = pd.Timestamp('2022-01-01') 
        end = pd.Timestamp('2022-05-10')
        func = utils(self.stock_code, start, end)
        # 获取对比数据
        time = func.getPrice().index
        raw_returns = get_price([self.stock_code], start_date=start, end_date=end, 
                                fre_step='1d', fields=['quote_rate'])[self.stock_code]['quote_rate'].to_list()
        normalized_returns = func.getNormalizedReturns(self.interval)
        
        # 绘图，曲线事先进行归一化来消除量纲
        plt.figure(figsize=(12,8))
        plt.plot(time, func.standardize(raw_returns), label='raw')
        plt.plot(time, func.standardize(normalized_returns), label='normalized')
        plt.legend(loc=0)
        plt.show()
        
    def runTest(self):
        self.stock_code = '300750.SZ'
        self.interval = 21
        # tests
        self.testNoneReturnSeries()
        self.testNormalSituation()

@Test
class testVolatility(unittest.TestCase):
    def runTest(self):
        ''' 直接作图查看
        '''
        import matplotlib.pyplot as plt
        # 请求宁德时代在 2022/1/1 -> 2022/4/1 的收盘价数据
        start = pd.Timestamp('2022-01-01')
        end = pd.Timestamp('2022-04-01')
        func = utils('300750.SZ', start, end)
        returns = func.getPrice()
        # 计算波动率
        vols = func.getVolRange(60, 0.94).tolist()
        time = get_trade_days(start, end).strftime('%m/%d')
        # 绘图，曲线事先进行归一化来消除量纲
        plt.figure(figsize=(12,8))
        plt.plot(time, func.standardize(returns), label='return')
        plt.plot(time, func.standardize(vols), label='vol')
        plt.legend(loc=0)
        plt.show()
