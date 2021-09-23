import numpy as np
from matplotlib import pyplot as plt
import cv2


def boxCounting_1(img):
    """
    计算二值图像计盒维数（像素点在盒子边缘）
    """
    cv2.imshow("img", img)
    cv2.imshow('o', o)
    h, w = img.shape[:2]

    Nr = []  # 盒子数列表

    for i in range(1, min(h // 2 + 1, w // 2 + 1)):  # 盒子边长从1到图片最小尺寸的二分之一

        box_num = 0  # 初始化盒子数
        for row in range(h // i):  # （h//i）为盒子的行数，（w//i）为盒子的列数
            for col in range(w // i):  # h,w为一个盒子立方体的边长
                if np.any(img[row * i:(row + 1) * i + 1, col * i:(col + 1) * i + 1] != 0):
                    #  上下两行都去掉，换成int[(max-min)/i]+1，是该区域A需要补充的盒子数量
                    #  建立新的图，在上面填充，可以用到上面的if里面的区域
                    box_num += 1
        Nr.append(box_num)
    return Nr, 1


def boxCounting_2(img):
    """
    计算二值图像计盒维数（像素点在盒子内部）
    """
    h, w = img.shape[:2]

    Nr = []  # 盒子数列表

    # 移动图像像素点到奇数行和列
    new_img = np.zeros((2 * h, 2 * w), np.uint8)
    for x in range(h):
        for y in range(w):
            new_img[2 * x + 1, 2 * y + 1] = img[x, y]

    cv2.imshow("new_img", new_img)
    cv2.imshow("img", img)
    cv2.imshow('o', o)
    print("new_img", new_img.shape, "img", img.shape)

    for i in range(2, min(h + 1, w + 1), 2):  # 盒子边长从2到图片最小尺寸的二分之一

        box_num = 0  # 初始化盒子数
        for row in range(2 * h // i):  # （2h//i）为盒子的行数，（2w//i）为盒子的列数
            for col in range(2 * w // i):
                if np.any(new_img[row * i:(row + 1) * i + 1, col * i:(col + 1) * i + 1] != 0):
                    box_num += 1
        Nr.append(box_num)
    return Nr, 2


def Least_squares(x, y):
    """
    输入x、y坐标集，用最小二乘法拟合曲线
    输出拟合直线y = ax+b 的参数a,b，和相关系数r
    """
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    l1 = np.zeros(1)
    l2 = np.zeros(1)
    l1_ = np.zeros(1)
    l2_ = np.zeros(1)
    r = np.zeros(1)

    for i in np.arange(len(x)):
        k = (x[i] - x_) * (y[i] - y_)
        m += k
        p = np.square(x[i] - x_)
        n = n + p
        l1 = np.square(x[i] - x_)
        l1_ = l1_ + l1
        l2 = np.square(y[i] - y_)
        l2_ = l2_ + l2

    a = m / n
    b = y_ - a * x_
    r = m / np.sqrt(l1_ * l2_)

    return a, b, r


if __name__ == "__main__":
    o = cv2.imread("./text_img/te.png", cv2.IMREAD_GRAYSCALE)
    thresh = cv2.threshold(o, 100, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('Original', thresh)
    # thresh = cv2.resize(thresh,None,fx=0.5,fy=0.5)
    # thresh = cv2.Canny(o,50,200)

    # 计算二值图像计盒维数
    Nr, interval = boxCounting_1(thresh)
    print('Nr', Nr)
    print('Nr_len', len(Nr))

    x = np.log(range(interval, interval * len(Nr) + interval, interval))
    y = np.log(Nr)

    # 最小二乘法拟合直线
    a, b, r = Least_squares(x, y)
    print(a, b, r * r)
    y1 = a * x + b

    # 设置微软雅黑，支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 新建一个figure对象
    plt.figure(figsize=(10, 5), facecolor='w')

    plt.plot(x, y, 'bo', lw=2, markersize=6, label='散点数据')
    plt.plot(x, y1, 'r-', lw=2, markersize=6, label='y = %.4fx+%.4f' % (a, b))
    plt.grid(b=True, ls=':')
    plt.xlabel(u'Ln[r]', fontsize=16)
    plt.ylabel(u'Ln[N(r)]', fontsize=16)
    plt.legend()  # 用来显示标签
    plt.text(0.1, 0.9, 'R^2 = %.4f' % float(r * r), fontsize=12, style="italic", weight="light",
             verticalalignment='center', horizontalalignment='right', rotation=0)
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()
