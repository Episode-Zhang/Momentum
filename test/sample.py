from packages.Test import Test
from unittest import TestCase


@Test
class testSample(TestCase):
    def testSum(self):
        my_sum = lambda list: list[-1] + my_sum(list[ : -1]) if len(list) > 0 else 0

        nums = [1, 2, 3, 4, 5]
        expected = sum(nums)
        actual = my_sum(nums)

        self.assertEquals(expected, actual)

    def testSumBuggy(self):
        my_sum = lambda list: my_sum(list[ : -1]) if len(list) > 0 else 0

        nums = [1, 2, 3, 4, 5]
        expected = sum(nums)
        actual = my_sum(nums)

        self.assertEquals(expected, actual)

    def runTest(self):
        # 应该没bug
        self.testSum()
        # 应该有bug，显示 15 != 10
        self.testSumBuggy()