##
# 对python标准库unittest进行了简单的封装
# 用于简单的白盒测试
# @author zzn
##

import unittest


def Test(tested_class: unittest.TestCase) -> None:
    ''' 测试类的装饰器(decorator)
        用于一键测试继承了"unittest.TestCase"类并实现了"runTest"方法的用户自建测试类
        使用时只需要在指定类前一行的开头添加 "@Test" 即可，类似于 junit4 框架的使用
        ! 注：请不要对该函数尝试除了 @Test 以外的用法！ !
    
    Args:
        tested_class: 需要测试的类
    '''
    # 类型检查
    if (not tested_class.__base__ is unittest.TestCase):
        raise TypeError('用户自定义的测试类需要单一继承"unittest.TestCase"类！')
    # 核心功能
    def do_test():
        suite = unittest.TestSuite()
        # 检查测试类是否实现了"runTest"方法
        if (hasattr(tested_class, 'runTest')):
            # 添加测试类
            suite.addTest(tested_class())
        else:
            raise AttributeError('测试类需要实现"runTest"方法，并将所有的测试写入该方法内')
        # 测试
        runner = unittest.TextTestRunner()
        runner.run(suite)
    return do_test()