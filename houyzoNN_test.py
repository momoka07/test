# 重みとバイアス初期値あり
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(filepath_or_buffer="sample2.csv", encoding="ms932", names=('y', 'x1', 'x2', 'x3'))
df.drop(index=1, axis=0)

df_corr = df.corr()
print(df_corr)
print(df_corr.loc['y','x1'])

sns.heatmap(df_corr, vmax=1, vmin=-1, center=0)
plt.savefig('seaborn_heatmap_corr_example.png')
# データシャッフル
df_s = df.sample(frac=1)

column = ['x1','x2','x3']

train_data=df_s[column]
train_labels=df_s['y']

# 重みとバイアスの初期値を算出
max = train_data.max(axis=0) 
min = train_data.min(axis=0)
a = [] #重みの初期値を入れる用
b = [] #バイアスの初期値を入れる用

for num in column:
    if df_corr.loc['y',num] >= 0:# 相関係数が正
        a.append(6 / (max[num]-min[num]))
        b.append(-6 * (min[num] + max[num]) / (2 * (max[num]-min[num])))
    elif df_corr.loc['y',num] < 0:# 相関係数が負
        a.append(-6 / (max[num]-min[num]))
        b.append(6 * (min[num] + max[num]) / (2 * (max[num]-min[num])))

print(a)