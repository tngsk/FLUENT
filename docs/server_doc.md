# Node.js Server (`node/server.js`) 解説

## 実装と理論的背景
`node/server.js` は、フロントエンド UI に対して API と静的ファイルを提供し、バックエンドの Python スクリプト群（Extractor, Segmenter, Trainer, Predictor など）を非同期に呼び出してオーケストレーションを行う Express サーバーです。

### 理論的背景
1. **非同期I/Oとイベント駆動アーキテクチャ**
Node.js はシングルスレッド・イベントループベースで動作するため、ファイルの読み書きや外部プロセスの実行などの重い処理でメインスレッドをブロックしないことが重要です。この実装では `fs/promises` や `child_process.spawn` を使用して、ブロッキングを回避しています。
2. **プロセス間通信 (IPC)**
Python で実装されたバックエンドとの連携には、標準入出力（`stdout` / `stderr`）とコマンドライン引数を使用しています。
3. **Server-Sent Events (SSE)**
機械学習の訓練プロセスのような、時間のかかる非同期処理の進捗状況をクライアント（ブラウザ）へリアルタイムにストリーミングするために使用されます。

### 実装解説
もとのコードの主要な処理部分を引用しながら解説します。

#### 非同期ファイルアクセス (`/api/labels` GET)
```javascript
// 保存済みラベルをすべて返す。まだ1件もなければ空の A-MAP 構造を返す
app.get("/api/labels", async (req, res) => {
  // subjectId に基づいてフィルタリング
  const labelsetPath = path.join(DATA_DIR, "labelset.json");
  try {
    const { subjectId } = req.query;
// ...
    await fs.access(labelsetPath, constants.F_OK);
    const fullLabelset = JSON.parse(await fs.readFile(labelsetPath, "utf8"));
```
`fs/promises` (ここでは `fs` としてインポートされていると想定) を使用して、ノンブロッキングで `labelset.json` を読み込みます。

#### Python プロセスの呼び出し
```javascript
const spawnPython = (script, args = []) =>
  new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [path.join(ROOT_DIR, script), ...args], {
      cwd: ROOT_DIR,
    });
    let out = "",
      err = "";
    proc.stdout.on("data", (d) => (out += d));
    proc.stderr.on("data", (d) => {
      err += d;
      console.error(`[${script}]`, d.toString().trim());
    });
    proc.on("close", (code) =>
      code === 0 || code === 2
        ? resolve({ out, code })
        : reject(new Error(err)),
    );
  });
```
`child_process.spawn` を用いて Python スクリプトを実行する汎用ヘルパー関数です。標準出力 `stdout` と標準エラー `stderr` を収集し、プロセス終了時の exit code によって Promise を resolve または reject します。exit code 2 は特定のエラー（例：モデル未存在）として特別に許容しています。

#### 学習進捗のストリーミング (SSE) (`/api/train` POST)
```javascript
  const proc = spawn(
    PYTHON_BIN,
    [path.join(ROOT_DIR, "python/train.py"), ...args],
    { cwd: ROOT_DIR },
  );
  trainingProcess = proc;

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");

// ...
  // train.py は各 epoch 後に JSON を1行 stdout へ出力する
  proc.stdout.on("data", (data) => {
    data
      .toString()
      .trim()
      .split("\n")
      .filter(Boolean)
      .forEach((line) => {
        try {
          JSON.parse(line);
          res.write(`data: ${line}\n\n`);
        } catch {}
      });
  });
```
学習を実行する `python/train.py` を `spawn` で起動し、HTTP のレスポンスヘッダに `text/event-stream` を設定します。Python スクリプトが標準出力に 1 行の JSON を吐き出すたびに、それをパース検証して SSE 形式（`data: ...\n\n`）でクライアントへ即座に送信（`res.write`）します。

#### パイプラインの連鎖呼び出し (`/api/tools/process`)
```javascript
    // Process successful, update extractor features and fit scaler
    const parsedOut = JSON.parse(out);
    if (parsedOut.success) {
      console.log("Audio processed successfully. Running feature extractor...");
      spawnPython("python/extractor.py")
        .then(() => console.log("Feature extractor completed."))
        .catch((err) => console.error("Feature extractor failed:", err));
    }
```
音声処理（セグメンテーションやクロップなど）が完了した後、データセット（WAVファイルの集合）に変更が生じたため、自動的に `python/extractor.py` を呼び出して特徴量の再抽出とスケーラーの再フィットを非同期で行っています。
