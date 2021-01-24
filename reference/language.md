# 文章を扱った論文・ライブラリ等まとめ

- [2018年](#2018年)
- [2019年](#2019年)

## 2018年

- [Top-Down Tree Structured Text Generation](https://arxiv.org/abs/1808.04865) (8月)
  - 文章を頭から逐次的に生成するのでは複雑な文章を構成できない
  - はじめに構文木を用いて文章構造を生成してから、品詞に対応する単語をうめていく
  - 構文木はRNNを使用して多段階で深くしていく
- Graph Convolutionを自然言語処理に応用する (2018年8月～2019年5月)
  - [Part.1](https://medium.com/programming-soda/graph-convolution%E3%82%92%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%81%AB%E5%BF%9C%E7%94%A8%E3%81%99%E3%82%8B-part1-b792d53c4c18):Graph Convolutionの歴史
  - [Part.2](https://medium.com/programming-soda/graph-convolution%E3%82%92%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%81%AB%E5%BF%9C%E7%94%A8%E3%81%99%E3%82%8B-part2-dd0f9bc25dd3):Graph Attention Networkの解説と実装
  - [Part.3](https://medium.com/programming-soda/graph-convolution%E3%81%A7%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%82%92%E8%A1%8C%E3%81%86-%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E5%88%86%E9%A1%9E%E7%B7%A8-part3-b85acee1a3e8):文章を係り受け解析or単語類似度に基づきグラフの構築・実験
  - [Part.4](https://medium.com/programming-soda/graph-convolution%E3%81%A7%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%82%92%E8%A1%8C%E3%81%86-%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E5%88%86%E9%A1%9E%E7%B7%A8-part4-caee203b86af):part.4の結果の分析した結果
  - [Part.5](https://medium.com/programming-soda/graph-convolution%E3%81%A7%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%82%92%E8%A1%8C%E3%81%86-%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88%E5%88%86%E9%A1%9E%E7%B7%A8-part5-end-cc9b0b4aac06):グラフ作成法の改善
  - [Part.6](https://medium.com/programming-soda/graph-convolution%E3%82%92%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%81%AB%E5%BF%9C%E7%94%A8%E3%81%99%E3%82%8B-part6-f4596b2bcc93):自然言語処理にGraph Convolutionを利用した論文まとめ
  - [Part.7](https://medium.com/programming-soda/graph-convolution%E3%82%92%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%81%AB%E5%BF%9C%E7%94%A8%E3%81%99%E3%82%8B-part7-end-3f6812ca08cf):まとめ
- [言語モデルの性能が、実装により異なる件を解決する](https://medium.com/programming-soda/%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E6%80%A7%E8%83%BD%E3%81%8C-%E5%AE%9F%E8%A3%85%E3%81%AB%E3%82%88%E3%82%8A%E7%95%B0%E3%81%AA%E3%82%8B%E4%BB%B6%E3%82%92%E8%A7%A3%E6%B1%BA%E3%81%99%E3%82%8B-5d36c841fcac) (10月)
  - Deep Learning フレームワークによって言語モデルの性能が異なるという話
  - ある系列長を与えると次の系列を予測する方法に比べて固定長系列を与える方式は精度が悪い
  - 逐次的に次の系列を予測させることで差異を減らすことができる

## 2019年

- [Stanza: A Python NLP Library for Many Human Languages](https://github.com/stanfordnlp/stanza) (1月)
  - 66言語(日本語含む)に対応した形態素解析・係り受け解析ライブラリ
  - neural networkベースでPyTorchで実装されている
  - 英語のみだが医学用語にも対応している
- [A Transformer Chatbot Tutorial with TensorFlow 2.0](https://medium.com/tensorflow/a-transformer-chatbot-tutorial-with-tensorflow-2-0-88bf59e66fe2) (3月)
  - TensorFlow 2.0のtf.kerasを使用してTransformerを実装するチュートリアル
  - Transformerベースのチャットボットを作成している
  - コードは約500行程度とシンプルになっている
  