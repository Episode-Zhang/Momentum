##
# 对 CPD 模块的简单的集成测试
# @author zzn
##

import unittest
import matplotlib.pyplot as plt
from src.CPD import CPD
from src.utils import utils
from packages.Test import Test
from mindgo_api import *


@Test
class testCPD(unittest.TestCase):
    def test1(self):
        # 请求中曼石油在 2021/8/1 -> 2022/4/6 的收盘价数据
        stock = '603619.SH'
        start = pd.Timestamp('2021-08-01')
        end = pd.Timestamp('2022-04-06')
        interval = 21
        time = get_trade_days(start, end).strftime('%m/%d')
        # 请求收盘价
        func = utils(stock, start, end)
        returns = func.getPrice()
        # 请求对应的cpd分数
        cpd_info = CPD().getCPDScoreAndLocation(stock, start, end, interval)
        cpd_score = [window[0][0] for window in cpd_info for i in range(len(window))]
        print([window[0][0] for window in cpd_info])
        # 绘图，曲线事先进行归一化来消除量纲
        plt.figure(figsize=(30,8))
        plt.plot(time, func.standardize(returns), label='return')
        plt.plot(time, func.standardize(cpd_score), label='cpd_score')
        plt.legend(loc=0)
        plt.show()
    
    def test2(self):
        # 请求比亚迪在 2021/10/1 -> 2022/4/6 的收盘价数据
        stock = '002594.SZ'
        start = pd.Timestamp('2021-10-01')
        end = pd.Timestamp('2022-04-06')
        interval = 21
        time = get_trade_days(start, end).strftime('%m/%d')
        # 请求收盘价
        func = utils(stock, start, end)
        returns = func.getPrice()
        # 请求对应的cpd分数
        cpd_info = CPD().getCPDScoreAndLocation(stock, start, end, interval)
        cpd_score = [window[0][0] for window in cpd_info for i in range(len(window))]
        print([window[0][0] for window in cpd_info])
        # 绘图，曲线事先进行归一化来消除量纲
        plt.figure(figsize=(30,8))
        plt.plot(time, func.standardize(returns), label='return')
        plt.plot(time, func.standardize(cpd_score), label='cpd_score')
        plt.legend(loc=0)
        plt.show()
    
    def runTest(self):
        ''' 直接作图查看
        '''
        self.test1()
        self.test2()