import imutils
import numpy as np
from matplotlib import pyplot as plt
import cv2


def boxCounting_1(img):
    """
    计算二值图像计盒维数（像素点在盒子边缘）
    """
    h, w = img.shape[:2]
    print(type(img))
    print(h, w, type(h))
    Nr = []  # 盒子数列表
    box_img = np.zeros((h, w))  # 初始化一个盒子维图
    # for i in range(1, min(h // 2 + 1, w // 2 + 1)):  # 盒子边长从1到图片最小尺寸的二分之一
    for i in range(1, 5):  # 盒子边长从1到图片最小尺寸的二分之一
        box_img = np.zeros((h, w))  # 初始化一个盒子维图
        box_num = 0  # 初始化盒子数
        for row in range(h // i):  # （h//i）为盒子的行数，（w//i）为盒子的列数
            for col in range(w // i):  # h,w为一个盒子立方体的边长
                max_g = img[row * i:(row + 1) * i + 1, col * i:(col + 1) * i + 1].max()  # 矩阵中最大值
                min_g = img[row * i:(row + 1) * i + 1, col * i:(col + 1) * i + 1].min()  # 矩阵中最小值
                # print(max_g, min_g)
                par_num = int((max_g - min_g) / i) + 1
                # print(par_num)
                box_num = par_num + box_num
                #  上下两行都去掉，换成int[(max-min)/i]+1，是该区域A需要补充的盒子数量
                box_img[row * i:(row + 1) * i + 1, col * i:(col + 1) * i + 1] = max_g
                #  建立新的图，在上面填充，可以用到上面的if里面的区域
        if i < 4:
            box_img = box_img.astype(np.uint8)
            cv2.imshow("box_img", box_img)
            cv2.waitKey(0)

        Nr.append(box_num)
    return Nr, 1


def improved(img):
    print(type(img), img)
    h, w = img.shape[:2]

    print(h, w, type(h))
    Nr = []  # 盒子数列表
    box_img = np.zeros((h, w))  # 初始化一个盒子维图
    # for i in range(1, min(h // 2 + 1, w // 2 + 1)):  # 盒子边长从1到图片最小尺寸的二分之一
    for i in range(1, 5):  # 盒子边长从1到图片最小尺寸的二分之一
        box_img = np.zeros((h, w))  # 初始化一个盒子维图
        box_num = 0  # 初始化盒子数
        for row in range(h // i):  # （h//i）为盒子的行数，（w//i）为盒子的列数
            for col in range(w // i):  # h,w为一个盒子立方体的边长
                temp_box = img[row * i:(row + 1) * i + 1, col * i:(col + 1) * i + 1]
                # print('temp_box', np.shape(temp_box))
                max_g = temp_box.max()  # 矩阵中最大值
                min_g = temp_box.min()  # 矩阵中最小值

                gamma = np.sum(temp_box - min_g) / ((max_g - min_g) * i * i)  # gamma值
                if gamma > 0.9:
                    par_num = 1
                    max_g = min_g
                else:
                    par_num = int((max_g - min_g) / i) + 1
                # print(par_num)
                box_num = par_num + box_num
                #  上下两行都去掉，换成int[(max-min)/i]+1，是该区域A需要补充的盒子数量
                box_img[row * i:(row + 1) * i + 1, col * i:(col + 1) * i + 1] = max_g
                #  建立新的图，在上面填充，可以用到上面的if里面的区域
        if i < 4:
            plt.imshow(box_img)
            plt.show()
            box_img = box_img.astype(np.uint8)
            cv2.imshow("box_img", box_img)
            cv2.waitKey(0)

        Nr.append(box_num)
    return Nr, 1


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


def run_box(o, func):
    # 计算二值图像计盒维数
    Nr, interval = func(o)
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


if __name__ == "__main__":
    o = cv2.imread("./text_img/thoughness/li.png", cv2.IMREAD_GRAYSCALE)
    # thresh = cv2.threshold(o, 100, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('Original', o)
    # plt.imshow(imutils.opencv2matplotlib(o))
    # plt.show()
    # thresh = cv2.resize(thresh,None,fx=0.5,fy=0.5)
    # thresh = cv2.Canny(o,50,200)

    # run_box(o, boxCounting_1)
    run_box(o, improved)
# Nr [30763292, 6307064, 2392323, 1203849]