# TensorflowTutorials

Tensorflowの公式チュートリアルを試した内容メモ

TensorFlow 1.3を使用

## 各チュートリアルの概要

### GettingStartedWithTensorflow.ipynb

Tensorflowの基礎を学ぶチュートリアル

- tensorflowのimport
- 計算グラフを作って，そして走らせる
    - `tf.constant()`は定数を表すノード
    - `tf.add()`のように演算もノード
    - `tf.placeholder()`は入力値などを入れる(動的に入れ替えられる)入れ物のような?ノード
    - 上記以外にも様々なノードがある．これらを組み合わせて計算グラフを作る
    - 作った計算グラフはSessionオブジェクトを作って`run()`することで実行される

### GettingStartedWithTensorflow_EstimatorBasicUsage.ipynb

高レベルなTensorflowライブラリである`tf.estimator`の使い方の基礎を学ぶチュートリアル
`tf.estimator.LinerRegressor()`を使って，簡単に線形回帰を試してみる．

- Numpyを使って配列を容易
    - 学習データ({`x_train`と`y_train`)を用意する
    - 評価データ(`x_eval`と`y_eval`)を用意する
- `tf.estimator.input.numpy_inpuf_fn()`を使うとNumpy型の配列を使って，関数オブジェクトを作れる
    - 学習用の関数オブジェクト(`input_fn`)
        - `x_train`と`y_train`を与える
        - 引数`num_epochs=None, shuffle=True`とするのが良い?
    - 学習データでの評価用関数オブジェクト(`input_fn_train`)
        - `x_train`と`y_train`を与える
        - 引数`num_epochs=1000, shuffle=False`とする
    - 評価データでの評価用関数オブジェクト(`input_fn_eval`)
        - `x_eval`と`y_eval`を与える
        - 引数`num_epochs=1000, shuffle=False`とする
    - (*) 評価の際に`num_epochs=1`としていないのは，確率的モデルを取り扱ったりする際に，複数回評価したいからぽい?
        - https://stackoverflow.com/questions/42816124/steps-vs-num-epochs-in-tensorflow-getting-started-tutorial
- `tf.estimator.train()`に学習用の関数オブジェクトを与えて，`steps=1000`の回数だけ学習する．
- `tf.estimator.evaluate()`に評価用の関数オブジェクトを与えると，評価結果が得られる．

### GettingStartedWithTensorflow_ACustomModel.ipynb


### GettingStartedWithTensorflow_LinearRegression.ipynb

