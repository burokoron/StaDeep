# その他雑多な論文・ライブラリ等まとめ

[2018年](#2018年)

## 2018年

- [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap) (2月)
  - ゲーム理論に基づいて機械学習モデルの判断根拠を可視化するライブラリ
  - ツリーアンサンブルライブラリはXGBoost/LightGBM/CatBoost/scikit-learn/pyspark modelsで使用可能
  - ディープラーニングモデルはTensorFlow/Keras/PyTorch modelsで使用可能
- [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) (3月)
  - 時系列データを扱えるCNNであるTemporal Convolutional Networks (TCN)を構築
  - 多くのデータセットでGRUを大幅に上回る
  - [実装](https://github.com/locuslab/TCN)
- [All-Optical Machine Learning Using Diffractive Deep Neural Networks](https://arxiv.org/abs/1804.08711) (4月)
  - 3Dプリンタで全結合NNと同じ働きをする穴だらけの板を作成した
  - 入力として画像を光で与えると分類クラスによって違う位置に光が出力される
  - MNISTで実験成功
- [The What-If Tool: Code-Free Probing of Machine Learning Models](https://ai.googleblog.com/2018/09/the-what-if-tool-code-free-probing-of.html) (9月)
  - 機械学習モデルの挙動を調査できるツール
  - 入力データを編集することで類似データなのに分類がことなる事象などを発見できる
  - kerasとscikit-learnで構築したモデルで使用可能
