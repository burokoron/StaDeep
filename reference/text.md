# 文章を扱った論文・ライブラリ等まとめ

[2018年](#2018年)

## 2018年

- [Top-Down Tree Structured Text Generation](https://arxiv.org/abs/1808.04865) (8月)
  - 文章を頭から逐次的に生成するのでは複雑な文章を構成できない
  - はじめに構文木を用いて文章構造を生成してから、品詞に対応する単語をうめていく
  - 構文木はRNNを使用して多段階で深くしていく
- Graph Convolutionを自然言語処理に応用する (8月～)
  - [Part.1](https://medium.com/programming-soda/graph-convolution%E3%82%92%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%81%AB%E5%BF%9C%E7%94%A8%E3%81%99%E3%82%8B-part1-b792d53c4c18):Graph Convolutionの歴史
  - [Part.2](https://medium.com/programming-soda/graph-convolution%E3%82%92%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%81%AB%E5%BF%9C%E7%94%A8%E3%81%99%E3%82%8B-part2-dd0f9bc25dd3):Graph Attention Networkの解説と実装
  - [Part.3](https://medium.com/programming-soda/graph-convolution%E3%81%A7%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%82%92%E8%A1%8C%E3%81%86-%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E5%88%86%E9%A1%9E%E7%B7%A8-part3-b85acee1a3e8):文章を係り受け解析or単語類似度に基づきグラフの構築・実験
  - [Part.4](https://medium.com/programming-soda/graph-convolution%E3%81%A7%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%82%92%E8%A1%8C%E3%81%86-%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E5%88%86%E9%A1%9E%E7%B7%A8-part4-caee203b86af):part.4の結果の分析した結果
  