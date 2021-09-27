import numpy as np
import cv2

arr = np.zeros((4, 4))
arr1 = np.zeros((4, 4))
arr2 = np.zeros((4, 4))
print(arr1, arr2)
arr1[1:4, 2:5] = 4
arr2[2:4, 1:5] = 8
print(arr1, arr2)
arr = arr1 - arr2
cv2.imshow('arr', arr)
print(type(arr), arr.size)
# print('qqq', type(arr[5][5]), arr[5][5])
cv2.waitKey(0)