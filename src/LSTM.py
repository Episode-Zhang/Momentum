##
# LSTM 神经网络模块
# @author zzn
##

import numpy as np
import shelve
import tensorflow.compat.v1 as tf


class LSTM_Model:
    ''' 接收特定股票在每一期的“标准化回报率”、“MACD因子值”、“CPD得分”、“CPD位置”，
        给出下一期该股票的持仓信号，来使得持仓资产的夏普率尽可能较高
        
    Attributes: 
          _model: LSTM_Model类的对象保存的具体模型
          _loss: 在训练对象持有的模型时所指定的损失函数
    '''
    ##
    # private
    # 以下方法均为类内私有方法，使用该类的用户不应该在类以外的任何地方调用这些方法
    ##
    def __SharpeLoss(self, y_true: np.ndarray, y_pred: tf.Tensor) -> tf.Tensor:
        ''' 模型默认的损失函数，定义为某一资产在一个时间窗口上的夏普比率
        
        parmas: y_true: 按损失函数的原意应为现实中实际观测到的y值；
                        在夏普损失函数中，y_true定义为，存储该段时间窗口上的资产“波动率”和“下一日回报率”的序列
                y_pred: 模型预测的下一期的持仓信号
                
        return: 由 y_true 和 y_pred 计算得到的损失函数值
        '''
        from numpy import sqrt
        # 初始化变量
        size = len(y_true)
        contributions = []
        target_sigma = 0.15
        # 对整个窗口上的数据计算资产的持仓贡献率R        
        for i in range(size):
            item = y_pred[i][0] * (target_sigma / y_true[i][0]) * y_true[i][1]
            contributions.append(item)
        # 计算贡献率的夏普率的负数
        sharpe_ratio = sqrt(252) * tf.reduce_mean(contributions) / tf.math.reduce_std(contributions)
        return -sharpe_ratio
    ##
    # public
    # 以下方法为公有方法，使用该类的用户仅可以在类外调用该类的公有方法
    ##
    def __init__(self):
        ''' 类的构造方法不接收任何参数，即表明类的初始化需要显式调用其它方法来完成
        '''
        self._model = None
        self._loss = None
        self._hasTrained = False
    
    def load(self, loc: str) -> None:
        ''' 从指定路径读入并加载模型
        
        params: loc: 加载模型的路径
        '''
        try:
            with shelve.open(loc) as model_db:
                # 数据读取
                info = model_db['basic_view']
                weights = model_db['weights']
                # 加载模型的结构和参数
                self._model = tf.keras.models.model_from_json(info)
                self._model.set_weights(weights)
                # 设置模型已训练
                self._hasTrained = True
        except Exception as e:
            log.info('加载模型失败，具体信息为：\n')
            log.info(e)
    
    def initialize(self, input_shape: tuple, output_shape: int, dropout_rate: float = 0.2) -> None:
        ''' 用于初始化一个将输出分量映射到(0,1)上的 LSTM 神经网络模型
        
        params: input_shape: 输入数据的维度
                output_shape: 输出数据的维度
                dropout_rate: 设置Dropout层中对每次训练一个batch后模型参数随机遗忘的比例
        '''
        # 建模
        model = tf.keras.Sequential()
        # 添加隐层
        model.add(tf.keras.Input(shape=input_shape))
        model.add(tf.keras.layers.LSTM(output_shape))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        # 将模型作为类的私有成员
        self._model = model
    
    def setLossFunction(self, loss: callable) -> None:
        ''' 为模型设置 Loss 函数，默认采用类内预定义的__SharpeLoss，用户可以给出自定义的 Loss 函数，需要满足形如
                    "Tensor loss_func(Tensor/np.ndarray y_true, Tensor y_pred)"
            的样式，并使用 tensorflow 库提供的数学函数来实现 
            (否则可能会导致类型失配的bug，强制类型转换有可能带来tensorflow无法自适应地求梯度的bug)
        
        parmas: loss: 为模型设定的 Loss 函数
        '''
        if loss == 'default':
            self._loss = self.__SharpeLoss
        else:
            self._loss = loss
    
    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int, epochs: int) -> None:
        ''' 用于训练模型
        
        params: x: 输入数据
                y: 目标数据
                batch_size: 神经网络每次训练批次的大小
                epochs: 神经网络完整训练一次的轮数
        '''
        if self._model == None:
            raise RuntimeError('请先调用"initialize"方法初始化模型！')
        if self._loss == None:
            raise RuntimeError('请先调用"setLossFunction"方法为神经网络指定损失函数！')
        # 编译模型
        self._model.compile(optimizer='Adam', loss=self._loss, metrics=['accuracy'], run_eagerly=True)
        # 训练模型
        self._model.fit(x, y, batch_size, epochs, shuffle=False, validation_split=0.1)
        # 输出模型结果
        print('模型训练完毕')
        self._model.summary()
        self._hasTrained = True
        
    def predict(self, x: np.ndarray) -> float:
        ''' 给出一个时间点或者一个时间窗口上的数据，返回下一期的持仓信号
        
        params: x: 用于输入神经网络的数据
        
        return: 下一期的持仓信号
        '''
        if not self._hasTrained:
            raise RuntimeError('请先调用"fit"方法来训练模型！')
        signal_result = self._model.predict(x)
        return signal_result

    def save(self, loc: str) -> None:
        ''' 将模型保存到指定的路径
        
        params: loc: 模型保存的路径
        '''
        if not self._hasTrained:
            raise RuntimeError('请先调用"fit"方法来训练模型！')
        # 保存模型
        with shelve.open(loc) as model_db:
            # 模型的结构中需要去除 ragged 关键字，因为 tensorflow 1.x 不支持
            model_db['basic_view'] = self._model.to_json().replace('"ragged": false, ', '')
            model_db['weights'] = self._model.get_weights()
