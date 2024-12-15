"""
変数減少法（Backwardeliminationmethod）とは
1.有意水準を決める
・モデルの中で「この変数が本当に必要か」を判断する基準を設定する。
例えは、有意水準を0.05（5%）だとすると、p値が0.05を超える変数は重要ではないとみなします。

2.最初のモデルを作成する
・はじめは、全ての独立変数を含めたモデルを作成する。
これが出発点になります。

3.最も高いp値の変数を確認
・作成したモデルの中で、各変数がどれだけ影響力を持ちかを「p値」を使って確認する。
p値が高いほど、その変数はモデルに必要ない可能性があります。
最もp値が高い変数を探します。

4.p値が有意水準を超えるか確認
・p値が有意水準を超える場合: 
その変数は統計的に重要でないとみなし、モデルから除きます。
・p値が有意水準以下の場合:
その変数は重要だとみなします。
全ての変数のp値が有意水準以下になれば、この時点でモデルは完成します。

5.変数を除いてモデルを再作成
・重要でない変数を除いた状態で、再びモデルを作成します。
・この操作を繰り返し、全ての残りの変数が有意水準以下になるまで進めます。

最終的な結果
・モデルに必要な変数だけが残り、シンプルかつ効果的なモデルが完成する。

例
独立変数として「年齢」「収入」「教育レベル」「趣味」の４つを含むモデルを考える。
1.モデルを作成してp値を確認したところ、「趣味」のp値が0.2と有意水準0.05よりも高い。
2.「趣味」を除外して新しいモデルを作成。
3.次に「教育レベル」のp値が0.08で、これも除外する。
4.最終的に「年齢」と「収入」だけが残り、この２つがモデルにとって重要な変数となる。
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# データセットのインポート
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('-----データセットのインポート-----')
print(X)

# カテゴリ変数のエンコーディング
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print('-----カテゴリ変数のエンコーディング-----')
print(X)

# ダミー変数トラップを避ける
X = X[:, 1:]

# 訓練用とテスト用へのデータセットの分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 訓練用データセットを使った重回帰モデルの訓練
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 変数減少法を使った変数の抽出
print('-----変数減少法について------')
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())
