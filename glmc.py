import cv2
import numpy as np
from numba import jit

np.set_printoptions(suppress=True)  # 输出时禁止科学表示法，直接输出小数值


def glgcm(img_gray, ngrad=16, ngray=16):
    '''Gray Level-Gradient Co-occurrence Matrix,取归一化后的灰度值、梯度值分别为16、16'''
    # 利用sobel算子分别计算x-y方向上的梯度值
    gsx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gsy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    height, width = img_gray.shape
    grad = (gsx ** 2 + gsy ** 2) ** 0.5  # 计算梯度值
    grad = np.asarray(1.0 * grad * (ngrad - 1) / grad.max(), dtype=np.int16)
    gray = np.asarray(1.0 * img_gray * (ngray - 1) / img_gray.max(), dtype=np.int16)  # 0-255变换为0-15
    gray_grad = np.zeros([ngray, ngrad])  # 灰度梯度共生矩阵
    for i in range(height):
        for j in range(width):
            gray_value = gray[i][j]
            grad_value = grad[i][j]
            gray_grad[gray_value][grad_value] += 1
    gray_grad = 1.0 * gray_grad / (height * width)  # 归一化灰度梯度矩阵，减少计算量
    glgcm_features = get_glgcm_features(gray_grad)
    return glgcm_features



def get_glgcm_features(mat):
    '''根据灰度梯度共生矩阵计算纹理特征量，包括小梯度优势，大梯度优势，灰度分布不均匀性，梯度分布不均匀性，能量，灰度平均，梯度平均，
    灰度方差，梯度方差，相关，灰度熵，梯度熵，混合熵，惯性，逆差矩'''
    sum_mat = mat.sum()
    small_grads_dominance = big_grads_dominance = gray_asymmetry = grads_asymmetry = energy = gray_mean = grads_mean = 0
    gray_variance = grads_variance = corelation = gray_entropy = grads_entropy = entropy = inertia = differ_moment = 0
    for i in range(mat.shape[0]):
        gray_variance_temp = 0
        for j in range(mat.shape[1]):
            small_grads_dominance += mat[i][j] / ((j + 1) ** 2)
            big_grads_dominance += mat[i][j] * j ** 2
            energy += mat[i][j] ** 2
            if mat[i].sum() != 0:
                gray_entropy -= mat[i][j] * np.log(mat[i].sum())
            if mat[:, j].sum() != 0:
                grads_entropy -= mat[i][j] * np.log(mat[:, j].sum())
            if mat[i][j] != 0:
                entropy -= mat[i][j] * np.log(mat[i][j])
                inertia += (i - j) ** 2 * np.log(mat[i][j])
            differ_moment += mat[i][j] / (1 + (i - j) ** 2)
            gray_variance_temp += mat[i][j] ** 0.5

        gray_asymmetry += mat[i].sum() ** 2
        gray_mean += i * mat[i].sum() ** 2
        gray_variance += (i - gray_mean) ** 2 * gray_variance_temp
    for j in range(mat.shape[1]):
        grads_variance_temp = 0
        for i in range(mat.shape[0]):
            grads_variance_temp += mat[i][j] ** 0.5
        grads_asymmetry += mat[:, j].sum() ** 2
        grads_mean += j * mat[:, j].sum() ** 2
        grads_variance += (j - grads_mean) ** 2 * grads_variance_temp
    small_grads_dominance /= sum_mat
    big_grads_dominance /= sum_mat
    gray_asymmetry /= sum_mat
    grads_asymmetry /= sum_mat
    gray_variance = gray_variance ** 0.5
    grads_variance = grads_variance ** 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            corelation += (i - gray_mean) * (j - grads_mean) * mat[i][j]
    glgcm_features = [small_grads_dominance, big_grads_dominance, gray_asymmetry, grads_asymmetry, energy, gray_mean,
                      grads_mean,
                      gray_variance, grads_variance, corelation, gray_entropy, grads_entropy, entropy, inertia,
                      differ_moment]
    return np.round(glgcm_features, 4)


if __name__ == '__main__':
    fp = './text_img/thoughness/2.png'
    img = cv2.imread(fp)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glgcm_features = glgcm(img_gray, 15, 15)
    print(glgcm_features)
