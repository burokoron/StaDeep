# 強化学習に関する論文・ライブラリ等まとめ

[2018年](#2018年)

## 2018年

- [Reinforcement Learning with Prediction-Based Rewards](https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/) (10月)
  - 見たことない状態を入力とした固定ランダムネットワークの出力を予測することは難しいことを利用し、予測ができないほど報酬を高くすることで未知の状態へ探索しやすくする
  - Montezuma’s Revengeで初めて人間の平均スコアを上回る
  - [論文](https://arxiv.org/abs/1810.12894)、[実装](https://github.com/openai/random-network-distillation)
