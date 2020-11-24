# その他雑多な論文・ライブラリ等まとめ

[2018年](#2018年)
[2019年](#2019年)

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
- [PySyft](https://github.com/OpenMined/PySyft) (11月)
  - フェデレーテッドラーニング、差分プライバシー、暗号化計算がライブラリ
  - PyTorchやTensorFlowなどで使用可能
  - [論文](https://arxiv.org/abs/1811.04017)
- [pytorchをさくっと学べるチュートリアル](https://github.com/pukkapies/dl-imperial-maths/tree/master/pytorch-tutorial) (11月)
  - [Part.0](https://github.com/pukkapies/dl-imperial-maths/blob/master/pytorch-tutorial/0.%20Computation%20Graphs.ipynb)：Computation Graphs
    - 計算グラフの構築
    - 入力を表示し、Webからランダムな猫画像を取得・表示する
  - [Part.1](https://github.com/pukkapies/dl-imperial-maths/blob/master/pytorch-tutorial/1.%20Object%20Classification%20with%20CNNs.ipynb)：Object Classification with CNNs
    - CIFAR-10を用いて画像分類
    - 単純な6層CNNを構築してCPUで学習
    - ResNeXt14を構築してGPUで学習
  - [Part.2](https://github.com/pukkapies/dl-imperial-maths/blob/master/pytorch-tutorial/2.%20Question%20Answering%20with%20RNNs.ipynb)：Question Answering with RNNs
    - RNNを使用したQAタスクを解くネットワークの構築
    - bAbIデータセットを用いる
    - Attentionを利用した判断根拠の可視化
  - [Part.3](https://github.com/pukkapies/dl-imperial-maths/blob/master/pytorch-tutorial/3.%20Saliency%20Maps%2C%20Feature%20Visualisation%20and%20Adversarial%20Examples.ipynb)：Saliency Maps, Feature Visualisation and Adversarial Examples
    - 画像分類における判断基準の可視化
    - 画像のスタイル変換
    - 画像の敵対的攻撃
  - [Part.4](https://github.com/pukkapies/dl-imperial-maths/blob/master/pytorch-tutorial/4.%20Generative%20Modelling%20with%20VAEs%2C%20GANs%20and%20Autoregressive%20Models.ipynb)：Generative Modelling with VAEs, GANs and Autoregressive Models
    - 画像生成モデルの構築・学習
    - VAE, GAN, 自己回帰モデルの構築・学習
    - それぞれの生成モデルが生成する画像の違いを確認
  - [Part.5](https://github.com/pukkapies/dl-imperial-maths/blob/master/pytorch-tutorial/5.%20Discrete%20and%20Continuous%20Control%20with%20A2C%20and%20PPO.ipynb)：
Discrete and Continuous Control with A2C and PPO
    - OpenAI Gymを使って強化学習を行う
    - ゲームはCartPoleとPendulumを対象とする
    - 強化学習アルゴリズムのA2CとPPOを実装して実験
- [WaveRNN + VQ-VAE](https://github.com/mkotha/WaveRNN) (11月)
  - VQ-VAEとWaveRNNで音声話者変換を行う
  - データセットにVCTKを用い、PyTorchで実装されている
  - どちらのネットワークも論文ままだと上手くいかなかったため、変更が加えられている

## 2019年

- [TSNE-CUDA](https://github.com/CannyLab/tsne-cuda) (2月)
  - GPUで動くT-SNEライブラリ
  - scikit-learnより1200倍高速
  - 使い方はscikit-learnと同じ
  