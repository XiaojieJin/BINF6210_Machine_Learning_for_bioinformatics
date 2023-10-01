import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_pca import PCA

def scree_plot(result):
    a = result.percant_variance_explained
    x = [i for i in range(1,len(a)+1)]
    y = a

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    plt.tick_params(width = 3)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.scatter(x,y,color="#4169E1",linewidth=3,alpha = 0.8)


    plt.xlabel("Index of PC",fontsize = 28)
    plt.ylabel(f"Percentage",fontsize = 28)
    plt.tight_layout()
    plt.title("Scree Plot",fontsize = 30)
    plt
    
def score_plot(result):
    a = result.score
    x = a[:,0]
    y = a[:,1]

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    plt.tick_params(width = 3)
    plt.xticks(fontsize=22, rotation = 45)
    plt.yticks(fontsize=22)

    plt.scatter(x,y,color="#4169E1",linewidth=3,alpha = 0.8)


    plt.xlabel("PC1",fontsize = 28)
    plt.ylabel(f"PC2",fontsize = 28)
    plt.tight_layout()
    plt.title("Score Plot",fontsize = 30)
    plt

def loadings_plot(result):
    a = result.loadings
    x = a[:,0]
    y = a[:,1]

    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    plt.tick_params(width = 3)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.scatter(x,y,color="#4169E1",linewidth=3,alpha = 0.8)


    plt.xlabel("PC1",fontsize = 28)
    plt.ylabel(f"PC2",fontsize = 28)
    plt.tight_layout()
    plt.title("Loadings Plot",fontsize = 30)
    plt

hw2_prob3 = np.loadtxt("D:/personal/UNCC_file/BINF6201_ML/Homework_2_dataset_prob3.csv", delimiter = ",",
                  skiprows = 1)
# with open("D:/personal/UNCC_file/BINF6201_ML/Homework_2_dataset_prob4.csv") as f:
#     ncols = len(f.readline().split(','))

# hw2_prob4 = np.loadtxt("D:/personal/UNCC_file/BINF6201_ML/Homework_2_dataset_prob4.csv", delimiter = ",",
#                    skiprows = 1, usecols = range(1,ncols))
# hw2_prob4 = hw2_prob4.T
# a = hw2_prob4
# x = a[:,0]
# y = a[:,1]

# fig, ax = plt.subplots()
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_linewidth(3)
# ax.spines['left'].set_linewidth(3)
# plt.tick_params(width = 3)
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)

# plt.scatter(x,y,color="#4169E1",linewidth=3,alpha = 0.8)


# plt.xlabel("V1",fontsize = 28)
# plt.ylabel(f"V2",fontsize = 28)
# plt.tight_layout()
# plt.title("Raw data",fontsize = 30)
# plt
# array = np.array([[3080, 3180, 9080, 6880, 4180],
#                   [2460, 1960, 9480, 6010, 3460],
#                   [2720, 2820, 8920, 6120, 3320]])
obj = PCA(hw2_prob3)
obj_result = obj.fit_transform()
print(obj_result.percant_variance_explained)
# score_plot(obj_result)
# #print(np.matmul(obj.fit_transform()[1].T,obj.fit_transform()[1]))