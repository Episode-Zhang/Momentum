##
# CPD模型相关
# @ref https://arxiv.org/abs/2105.13727
# @author zzn
##

import warnings
import numpy as np
import pandas as pd
from  .utils import utils
from scipy.optimize import minimize
from mindgo_api import *


class CPD:
    ''' Change Point Detection模块，给定一个时间窗口上的资产回报率
        度量该时间窗口上change point的强度以及各点到change point的距离
        原理是给Gauss Process提供对应的核函数来拟合金融资产的价格波动，通过核函数的参数来捕捉和保存特征
    '''
    ##
    # protected
    # 以下方法均为类内保护方法，使用该类的用户不应该在类以外的任何地方调用这些方法
    ##
    def _sigmoidModel(self, location: float, steepness: float) -> callable:
        ''' 带参数的 sigmoid 函数，将其分离为两部分，一部分用来接收超参数，另一部分用来接收自变量
        
        Args:
            location: 参数
            steepness: 参数
        
        Returns:
            一个仅接受自变量x，并返回因变量值的 sigmoid 函数
        '''
        def sigmoid(x: float) -> float:
            ''' 仅接受自变量x的sigmoid函数
            
            Args:
                x: 自变量
            
            Returns:
                sigmoid值
            '''
            from numpy import exp
            return 1 / (1 + exp(-steepness * (x - location)))
        return sigmoid
    
    def _MaternKernelModel(self, λ: float, σ_h: float) -> callable:
        ''' 带参数的 Matern 3/2 核函数，将其分离为两部分，一部分用来接收超参数，另一部分用来接收自变量
        
        Args:
            λ: 超参数
            σ_h: 超参数
        
        Returns:
            一个仅接收自变量(x, x')，并返回因变量值的 kernel 函数
        '''
        def kernel(x1: float, x2: float) -> float:
            ''' 接受两个自变量，返回根据kernel的表达式计算得到的因变量值
            
            Args:
                x1: 自变量1
                x2: 自变量2
            
            Returns:
                根据表达式得出的值
            '''
            from numpy import sqrt, square, exp
            tmp = sqrt(3) * abs(x1 - x2) / λ
            return square(σ_h) * (1 + tmp) * exp(-tmp)
        return kernel    
        
    def _CPKernelModel(self, kernel1: callable, kernel2: callable, sigmoid: callable) -> callable:
        ''' 带参数的 Change Point 核函数，将其分离为两部分，一部分用来接收超参数，另一部分用来接收自变量
        
        Args:
            kernel1: 一个带超参数集合的核函数 
            kernel2: 另一个带超参数集合的核函数 
            sigmoid: 一个带超参数的sigmoid函数
                
        Returns:
            一个仅接收自变量(x, x')，并返回因变量值的 kernel 函数
        '''
        def kernel(x1: float, x2: float) -> float:
            ''' 接受两个自变量，返回根据kernel的表达式计算得到的因变量值
            
            Args:
                x1: 自变量1
                x2: 自变量2
            
            Returns:
                根据表达式得出的值
            '''
            return kernel1(x1, x2) * sigmoid(x1) * sigmoid(x2) + kernel2(x1, x2) * (1 - sigmoid(x1)) * (1 - sigmoid(x2))
        return kernel
    
    def _calcLikelihood(self, std_r: pd.Series, loc: np.array, interval: int, kernel: callable, σ_n: float) -> float:
        ''' 给定相关变量，计算 Matern kernel 参数 ξ 的似然估计
        
        Args:
            std_r: 标准化后的回报率序列
            loc: 时间窗口的位置点(横坐标)
            interval: 时间窗口的长度
            kernel: 进行参数估计的 Matern kernel
            σ_n: 随机绕动项服从的零均值正太分布的标准差
        
        Returns:
            给定参数和自变量下，Matern kernel 参数ξ的似然函数的值
        '''
        from numpy.linalg import inv, det
        from numpy import log, square, pi
        PI = pi
        # 部分数据类型转换成矩阵和向量参与运算
        if type(std_r) is not np.matrix:
            std_r = np.matrix(std_r)
        if type(loc) is not np.array:
            loc = np.array(loc)
        # 计算协方差矩阵
        n = std_r.size
        cov_mat = np.zeros(shape=(n, n))
        # 利用对称性，kernel(x1, x2) = kernel(x2, x1)，所以最后得到的是对称方阵
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    cov_mat[i][j] = kernel(loc[i], loc[j]) + square(σ_n)
                else:
                    cov_mat[i][j] = cov_mat[j][i] = kernel(loc[i], loc[j])
        cov_mat = np.matrix(cov_mat)
        # 计算似然函数值
        likelihood_value = (
            0.5 * (std_r * inv(cov_mat) * std_r.T) + 
            0.5 * log(det(cov_mat)) + 
            (interval + 1) / 2 * log(2 * PI))
        # 计时结束
        return likelihood_value.item(0)
    
    def _likelihoodModel(self, std_r: pd.Series, loc: np.array, interval: int, mode: str) -> callable:
        ''' 给定自变量并指定kernel的种类后，返回一个单独将参数ξ作为接口的似然函数
        
        Args:
            std_r: 标准化后的回报率序列
            loc: 时间窗口的位置点(横坐标)
            interval: 时间窗口的长度
            mode: kernel的种类，当前仅支持 "Matern"/"CP" 两种
        
        Returns:
            一个单独将参数ξ作为接口的极大似然函数
        '''
        # mode值为 Matern，则返回具有 Matern kernel 参数接口的似然函数
        def MaternLikelihood(ξ: list) -> float:
            ''' 接收 Matern kernel 的参数ξ，并返回相应的似然函数值
            
            Args:
                ξ: Matern kernel内含的参数，包括 λ，σ_h，σ_n
            
            Returns:
                给定参数 ξ 后计算得到的似然函数值
            '''
            # 检验参数个数是否达标
            if (len(ξ)) != 3:
                raise ValueError("参数个数不足，Matern kernel需要3个参数")
            λ, σ_h, σ_n = ξ[0], ξ[1], ξ[2]
            # 初始化kernel，计算值
            kernel = self._MaternKernelModel(λ, σ_h)
            likelihood_value = self._calcLikelihood(std_r, loc, interval, kernel, σ_n)
            return likelihood_value
        # mode值为 CP，则返回具有 ChangePoint kernel 参数接口的似然函数
        def CPLikelihood(ξ: list) -> float:
            ''' 接收 Change Point kernel 的参数ξ，并返回相应的似然函数值
            
            Args:
                ξ: Matern kernel内含的参数，包括 λ1，σ_h1，λ2，σ_h2，σ_n，location，steepness
            
            Returns:
                给定参数 ξ 后计算得到的似然函数值
            '''
            # 检验参数个数是否达标
            if (len(ξ)) != 7:
                raise ValueError("参数个数不足，CP kernel需要7个参数")
            λ1, σ_h1, λ2, σ_h2, σ_n = ξ[0], ξ[1], ξ[2], ξ[3], ξ[4]
            location, steepness = ξ[5], ξ[6]
            # 初始化kernel，计算值
            Matern_kernel1 = self._MaternKernelModel(λ1, σ_h1)
            Matern_kernel2 = self._MaternKernelModel(λ2, σ_h2)
            sigmoid = self._sigmoidModel(location, steepness)
            CPkernel = self._CPKernelModel(Matern_kernel1, Matern_kernel2, sigmoid)
            likelihood_value = self._calcLikelihood(std_r, loc, interval, CPkernel, σ_n)
            return likelihood_value
        # 根据mode选择方法行为
        if mode == 'Matern':
            return MaternLikelihood
        elif mode == 'CP':
            return CPLikelihood
        else:
            raise ValueError('未知的mode，请在"Matern"或"CP"中进行选择')
            
    def _getMaximumLikelihood(self, likelihood_func: callable, init_value: list, params_bounds: tuple) -> dict:
        ''' 给定一个接收参数的似然函数，以及参数的初值，返回参数的极大似然估计以及此时的似然函数值
        
        Args:
            likelihood_func: 需要优化的似然函数
            init_value: 优化的初始值，需要和似然函数接收的参数个数匹配
            params_bounds: 参数求解时迭代的上下界
        
        Returns:
            参数的极大似然估计以及似然函数值(如果可以求解得到)
        '''
        from scipy.optimize import minimize
        methods = ['L-BFGS-B', 'Powell', 'Nelder-Mead']
        # 默认使用 L-BFGS-B 算法求解，若失败则采用剩下两种算法
        for algorithm in methods:
            optimization = minimize(likelihood_func, init_value, method=algorithm, bounds=params_bounds)
            # 求解失败
            if not optimization.success:
                warnings.warn('优化失败！具体信息为：\n%s'%(optimization.message),
                              UserWarning, stacklevel=1)
                # 进行新的尝试，当且仅当还有剩余算法可以尝试
                if algorithm != methods[-1]:
                    log.info('现在尝试切换优化算法来进行求解')
                    continue
            break
        # 如果三个算法均导出了失败的优化，那么根据上面的循环，变量optimization会保存最后一次的结果
        info = {
            'func_value': optimization.fun, 
            "param_value": optimization.x}
        return info
    
    def _divideIntoSegments(self, std_returns: pd.Series, interval: int) -> tuple:
        ''' 接受输入的时序的标准化回报率序列，以及窗口长度interval，返回划分好的标准化价格段以及位置段
        
        Args:
            std_returns: 标准化的回报率序列
            interval: 时间窗口的长度
        
        return:
            划分好的标准化价格段以及位置段元组，具体形式为：(标准化价格段, 位置段)
        '''
        # 查看回报率序列是否为空
        if std_returns.size == 0:
            raise ValueError('输入的回报率序列不能为空！')
        # 初始化返回值
        std_returns_segments = []
        locations_segments = []
        # 计算段数以及余数
        segments_number = std_returns.size // interval
        remainder = std_returns.size % interval
        # 根据interval的值划分时间窗口
        for i in range(segments_number):
            start_loc = i * interval
            end_loc = (i + 1) * interval
            # 切分
            std_returns_segments.append(std_returns[start_loc: end_loc])
            # 每一个窗口分段的位置都从0开始计数
            locations_segments.append([i for i in range(interval)])
        # 余数补足
        if remainder > 0:
            std_returns_segments.append(std_returns[-remainder: ])
            locations_segments.append([i for i in range(remainder)])
        return (std_returns_segments, locations_segments)
    
    def _getCPDScoreAndLocationOnce(self, std_returns: pd.Series, locations: list) -> list:
        ''' 输入一个时间窗口的标准化的回报率序列以及位置序列，返回相应的窗口上的CPD score 和 location
            注：CPD score 特征是针对“一个窗口”的，在固定的窗口期上每天都相等
               CPD location 是针对每一天的，在固定的窗口期上每天各不相等
        
        Args:
            std_returns: 一个时间窗口上的回报率序列
            locations: 一个时间窗口上的位置序列
        
        Returns:
            一个时间窗口上的CPD score 和 location
        '''
        from numpy import exp
        # 校验时间窗口期是否唯一
        if std_returns.size != len(locations):
            raise ValueError('请检查给定参数的长度是否相等，该方法要求给定的参数长度相等！')
        # 初始化不同 kernel 的似然函数
        window_length = std_returns.size
        Matern_likelihood = self._likelihoodModel(std_returns, locations, window_length, 'Matern')
        CP_likelihood = self._likelihoodModel(std_returns, locations, window_length, 'CP')
        # 给定参数的初值和上下界
        Matern_init_value, Matern_bounds = [1, 1, 1], ((None, None), (None, None), (None, None))
        CP_init_value = [1 if i != 5 else window_length // 2 for i in range(7)]
        CP_bounds = tuple([(None, None) if i != 5 else (0, window_length) for i in range(7)])
        # 求解 Matern kernel 参数的极大似然值
        try:
            Matern_likelihood_info = self._getMaximumLikelihood(Matern_likelihood, Matern_init_value, Matern_bounds)
        except BaseException as e:
            log.info('Matern kernel参数的似然函数计算错误，具体形式为：')
            log.info(e)
            raise RuntimeError("优化参数似然函数失败！")
        Matern_likelihood_value = Matern_likelihood_info['func_value']
        # 求解 CP kernel 参数的极大似然值
        try:
            CP_likelihood_info = self._getMaximumLikelihood(CP_likelihood, CP_init_value, CP_bounds)
        except BaseException as e:
            log.info('Change Point kernel参数的似然函数计算错误，具体形式为：')
            log.info(e)
            raise RuntimeError("优化参数似然函数失败！")
        CP_likelihood_value = CP_likelihood_info['func_value']
        CP_location = CP_likelihood_info['param_value'][-2]
        # 计算 CP score 和窗口内各点到 location 的距离
        CP_score = 1 - 1 / (1 + exp(-(CP_likelihood_value - Matern_likelihood_value)))
        distance2CP = lambda loc: abs(CP_location - loc) / window_length
        CPD_info = [(CP_score, distance2CP(loc)) for loc in locations]
        return CPD_info
    ##
    # public
    # 以下方法为公有方法，使用该类的用户仅可以在类外调用该类的公有方法
    ##
    def getCPDScoreAndLocation(self, stock_code: str, start_date: pd.Timestamp, end_date: pd.Timestamp, interval: int) -> list:
        ''' 给定股票代码、起始日期以及相应的时间窗口，返回所有窗口内该股票拐点强弱信号以及窗口内每个点距离拐点的距离
            注：一个时间窗口内的拐点强弱信号处处相等，衡量的是『该窗口内』是否有出现“值得警惕和注意”的拐点；
               而当前点距离窗口内拐点的距离是互异的，这个特征是想对于点来说的
        
        Args:
            stock_code: 需要考察的股票代码
            start_date: 考察的起始日期
            end_date: 考察的结束日期
            interval: 时间窗口的跨度
        
        Returns:
            该股票的时序数据，每一日在对应时间窗口内具有的拐点强弱信号以及当日距所属时间窗口内拐点的距离
        '''
        # interval 必须为正整数
        if not type(interval) is int or interval <= 0:
            raise ValueError('参数interval必须为正整数！')
        # 获取给定日期间的收盘价并标准化
        CPD_utils = utils(stock_code, start_date, end_date)
        all_std_returns = CPD_utils.standardize(CPD_utils.getPrice())
        # 获取划分好的资产回报值窗口以及位置窗口
        returns_by_windows, location_by_windows = self._divideIntoSegments(all_std_returns, interval)
        # 滑动窗口求解不同窗口上的 CP score 和 location
        CPD_score_location_by_windows = []
        size = len(returns_by_windows)
        for window in range(size):
            window_info = self._getCPDScoreAndLocationOnce(returns_by_windows[window], location_by_windows[window])
            CPD_score_location_by_windows.append(window_info)
        return CPD_score_location_by_windows
