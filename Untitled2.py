#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from ev import OptimalSubsetRegression

# 加载数据
x = np.loadtxt("prostate/x.txt", delimiter=",")
y = np.loadtxt("prostate/y.txt", delimiter=",")
index = np.loadtxt("prostate/index.txt", delimiter=",", dtype=bool)
names = np.loadtxt("prostate/names.txt", delimiter=",", dtype=str)

# 创建回归模型实例
regression = OptimalSubsetRegression(x, y, index, names, K=10)

# 训练模型
regression.train()

# 评估模型
regression.evaluate()

# 进行交叉验证
regression.cross_validate()

# 获取最佳模型
best_b_0, best_b_1, best_features = regression.best_model()

# 打印最佳模型信息
print(f"Best Model: b_0={best_b_0}, b_1={best_b_1}, Variables={best_features}")


# In[ ]:




