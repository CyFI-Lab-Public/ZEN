import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
'''
    YoloV5, YoloV7, YoloV8, Resnet, mobilenet llama nanogpt
HIC
KD
C3
GAM
Gold
sPd
res-ghost
res-se
mobilenet-ghost
mobilenet-3
llama-pt
llama 3
gptlora
gptRKW
'''
data = [[0.94, 0.25, 0.26, 0.09, 0.1, 0.1, 0.09],
        [0.97, 0.2, 0.26, 0.06, 0.08, 0.08, 0.07],
        [0.35, 0.89, 0.18, 0.07, 0.08, 0.08, 0.07],
        [0.32, 0.88, 0.18, 0.06, 0.08, 0.08, 0.07],
        [0.19, 0.06, 0.78, 0.16, 0.16, 0.16, 0.08],
        [0.26, 0.05, 0.59, 0.12, 0.16, 0.14, 0.06],
        [0.12, 0.08, 0.15, 0.53, 0.2, 0.17, 0.08],
        [0.08, 0.06, 0.13, 0.49, 0.19, 0.15, 0.05],
        [0.08, 0.04, 0.12, 0.13, 0.65, 0.1, 0.02],
        [0.15, 0.1, 0.17, 0.16, 0.51, 0.18, 0.1],
        [0.16, 0.13, 0.14, 0.12, 0.15, 0.71, 0.13],
        [0.14, 0.1, 0.13, 0.12, 0.13, 0.72, 0.11],
        [0.15, 0.15, 0.2, 0.22, 0.24, 0.15, 0.6],
        [0.15, 0.15, 0.19, 0.23, 0.22, 0.15, 0.75]]
data = np.asarray(data)
print(data.shape)
df = pd.DataFrame(data, columns=['YV5', 'YV7', 'YV8', 'RN', 'MN-V2', 'L2', 'nGPT'])
print(df)
# displaying the plotted heatmap
font = {'fontname':'Times New Roman'}
yticklabels = ['YV5-HIC','YV5-KD', 'YV7-C3','YV7-GAM', 'YV8-GD', 'YV8-SPD',
               'RN-GH', 'RN-SE', 'MNV2-GH', 'MNV3', 'L2-PT', 'L3', 'nGPT-LR', 'nGPT-GH']
# the content of labels of these yticks
xticklabels = ['YV5', 'YV7', 'YV8', 'RN', 'MN-V2', 'L2', 'nGPT']

hm = sns.heatmap(data=data, annot=True, fmt='.2f', cmap="crest", xticklabels=xticklabels, yticklabels=yticklabels)

plt.title('Model Similarity Heatmap',**font)
plt.xticks(fontname='Times New Roman')
plt.yticks(fontname='Times New Roman')
plt.show()