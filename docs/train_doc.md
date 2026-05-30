# Trainer (`python/train.py`) 解説

## 実装と理論的背景
`python/train.py` は、抽出された物理特徴量 (`dataset.json`) を入力（X）とし、ユーザーが付与した感性ラベル (`labelset.json`) を教師データ（Y）として、両者をマッピングする Multi-Layer Perceptron (MLP) モデルを訓練するスクリプトです。

### 理論的背景
1. **Multi-Layer Perceptron (MLP)**
複数の隠れ層を持つフィードフォワードニューラルネットワークです。ここでは `scikit-learn` の `MLPRegressor` を使用し、非線形なマッピングを学習します。
2. **正則化 (Regularization)**
過学習（Overfitting）を防ぐため、L2正則化（`alpha` パラメータ）を使用します。データセットが小さい場合、過学習しやすいため重要な役割を果たします。
3. **最適化アルゴリズム (Solver)**
データサイズに応じて最適化手法を切り替えています。サンプル数が少ない場合は準ニュートン法の一種である `L-BFGS` が安定して収束しやすく、多い場合は確率的勾配降下法に基づく `Adam` を使用します。

### 実装解説
もとのコードの主要な処理部分を引用しながら解説します。

#### データの結合と前処理
```python
    # dataset.json の ID と labelset.json の ID が一致するセグメントのみを訓練に使用する
    # ※ dataset.json (X) のキーは直接データが配列だが、
    #    labelset.json (Y) は subjectId ごとにネストされているため注意
```
入力と教師データは別々の JSON ファイルとして提供されるため、セグメント ID をキーとして結合 (INNER JOIN 相当) を行います。また、`labelset.json` は複数の被験者（Subject）のデータを持つ入れ子構造のため、これを適切に展開・平均化または特定被験者のデータを抽出する処理が行われます。

#### モデルの初期化とハイパーパラメータの設定
```python
        # 小規模データセット（サンプル数 < 20）の場合、lbfgs の方が数値的に安定し、かつ収束が速い
        solver = "lbfgs" if len(x_arr) < 20 else "adam"

        mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver=solver,
            alpha=alpha,
            max_iter=1000,
            random_state=42,
            early_stopping=False,
            tol=1e-5,
            n_iter_no_change=10,
            verbose=True,
        )
```
- `hidden_layer_sizes=(32, 16)`: 隠れ層は2層で、ユニット数はそれぞれ32、16です。入力（26次元）から出力（10次元）へと段階的に次元を変換しつつ、モデルが複雑になりすぎない（過学習を防ぐ）サイズ設定です。
- `activation="relu"`: 一般的な活性化関数である ReLU を使用。
- `solver`: 前述の通り、データ数に応じて `lbfgs` と `adam` を動的に選択。
- `alpha`: L2正則化の強度。コマンドライン引数で調整可能。
- `verbose=True`: 訓練の進捗（エポックごとの損失）を標準出力に出力させる設定。これを後段でキャプチャします。

#### 標準出力のキャプチャと進捗のストリーミング
```python
    # scikit-learn の C 実装ループを活用し高速化
    # verbose=True の stdout をキャプチャし、UI 用に JSON で出力する
    class VerboseCapture:
        def __init__(self, original_stdout):
            self.original_stdout = original_stdout
            # ...

        def write(self, text):
            if text.startswith("Iteration "):
                # 例: "Iteration 1, loss = 0.73030389"
                try:
                    parts = text.split(", loss = ")
                    epoch = int(parts[0].replace("Iteration ", "")) - 1 # UI 用に 0-indexed に変換
                    loss = float(parts[1].strip())
                    self.original_stdout.write(json.dumps({"epoch": epoch, "loss": loss}) + "\n")
                    self.original_stdout.flush()
                except Exception:
                    pass
```
Node.js サーバー (Express) 経由で UI にリアルタイムな学習進捗（SSE: Server-Sent Events）を伝えるため、`scikit-learn` が標準出力に出すテキストログを `VerboseCapture` クラスでキャプチャし、パースして JSON 形式に変換した上で再度標準出力へ書き出しています。

#### 訓練の実行と保存
```python
    try:
        mlp.fit(x_arr, y_arr)
    # ...
    # ============================================================
    # Step 11: 学習済みモデルを保存
    # ============================================================
    with open(model_path, "wb") as f:
        pickle.dump(mlp, f)
```
`mlp.fit()` により訓練が実行され、完了したモデルは Python 標準のシリアライズ形式である `pickle` を用いて保存されます。
