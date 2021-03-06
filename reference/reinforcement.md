# 強化学習に関する論文・ライブラリ等まとめ

- [2018年](#2018年)
- [2019年](#2019年)

## 2018年

- [Reinforcement Learning with Prediction-Based Rewards](https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/) (10月)
  - 見たことない状態を入力とした固定ランダムネットワークの出力を予測することは難しいことを利用し、予測ができないほど報酬を高くすることで未知の状態へ探索しやすくする
  - Montezuma’s Revengeで初めて人間の平均スコアを上回る
  - [論文](https://arxiv.org/abs/1810.12894)、[実装](https://github.com/openai/random-network-distillation)

## 2019年

- [Open-ended Learning in Symmetric Zero-sum Games](https://arxiv.org/abs/1901.08106) (1月)
  - チェスなどのゼロサムゲームのAI性能は対戦相手との相対的強さで評価される
  - このようなゲームは相性問題により三すくみ関係が発生し真の強さ評価が難しい
  - ナッシュ均衡に基づくPSROrNを提案し、 BlottoおよびDifferentiable Lottで高性能を達成
- [Go-Explore: a New Approach for Hard-Exploration Problems](https://arxiv.org/abs/1901.10995) (1月)
  - 探索系のゲームであるMontezuma’s RevengeとPitfallで人間のエキスパートを超えた手法
  - 以前の探索を記憶しておき、再スタート時には記憶をもとに以前の状況まで戻ってくることで効率化している
  - 模倣学習も組み込んでおり、ゲーム以外のロボット工学などに応用できる可能性がある
  