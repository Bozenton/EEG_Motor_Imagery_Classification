# 将一长串数字分解为两个最近整数相乘

import numpy as np


def calc_hw(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return start, int(factor)


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False


if __name__ == '__main__':
    print(calc_hw(640))  # 250, 256
