# Module C 開発者ガイド

ようこそ。このドキュメントは `node/` ディレクトリを担当する開発者向けの入門資料です。
まずここを読めば、コードを読む前に全体像がつかめます。

---

## Module C の仕事

FLUENT は Python（音声解析・学習）と Node.js（UI・API）の2層構造になっています。
Module C はそのユーザー接点です。やることは3つです。

- 音声セグメントを**試聴**し、感性ラベルを**入力・保存**する
- 学習の**開始・中断・リセット**を操作し、進捗をリアルタイムで確認する
- 未ラベルのセグメントに対して AI の**予測結果をプレビュー**する

```
ブラウザ (App.vue)
    ↕ HTTP / WebSocket
Express (server.js)
    ↕ child_process (stdout/stderr)
Python scripts (train.py, predict.py ...)
    ↕ ファイル読み書き
data/*.json / data/segments/*.wav
```

---

## まず動かす

```bash
# リポジトリルート（node/ の一つ上）で実行
cp .env.example .env      # 初回のみ。中身はほぼデフォルトで OK
npm --prefix node install
npm --prefix node run dev
```

ブラウザで `http://localhost:5173` を開くと UI が表示されます。

`npm run dev` は Vite（フロントエンド）と Express（API サーバー）を同時に起動します。
フロントエンドが `/api` や `/socket.io` にアクセスすると、Vite が自動的に
`localhost:3000`（Express）へ転送してくれるので、CORS を気にせず開発できます。

---

## ファイル構成

```
node/
├── server.js        ← API サーバー。Python との橋渡し役
├── src/
│   ├── App.vue      ← UI のすべて（現在は1ファイルに集約）
│   ├── main.js      ← Vue の起動エントリー（基本触らない）
│   └── style.css    ← Tailwind の読み込みだけ（基本触らない）
├── vite.config.js   ← プロキシ設定が書いてある
└── package.json
```

開発のほとんどは `server.js` か `App.vue` のどちらかを触ることになります。

---

## 環境変数（`../.env`）

```
PORT=3000                              # Express のポート
DATA_DIR=data                          # data/ フォルダへの相対パス
PYTHON_PATH=.venv/bin/python           # Python の実行パス
ALLOWED_ORIGINS=http://localhost:5173  # CORS 許可オリジン
```

すべてデフォルト値があるので、基本的に変更不要です。

---

## API 一覧（server.js）

| メソッド | パス | 何をするか |
|---|---|---|
| GET | `/api/config` | `config.json` を返す（フォームの定義） |
| GET | `/api/segments` | セグメント ID の一覧を返す |
| GET | `/api/labels` | `labelset.json` を返す |
| POST | `/api/labels` | ラベルを1件保存（自動バックアップあり） |
| POST | `/api/predict` | 指定セグメントの AI 予測値を返す |
| POST | `/api/train` | 学習を開始する（`resume: true` で再開） |
| POST | `/api/train/stop` | 学習を中断する（途中のモデルを保存して終了） |
| DELETE | `/api/model` | モデルと学習履歴を削除（リセット） |
| GET | `/api/train/meta` | 最終学習の日時と対象セグメント一覧を返す |

### Python を呼び出すときのルール

Python スクリプトを起動するときは `spawnPython` ヘルパーを使ってください。

```js
const { out, code } = await spawnPython("python/predict.py", ["--arg", "value"])
```

- exit code `0` → 成功
- exit code `2` → モデル未存在（エラーではなく「まだ学習していない」状態）
- それ以外 → reject（エラー扱い）

学習（`train.py`）だけは進捗をリアルタイムでストリームする必要があるため、
`spawnPython` を使わず `spawn` を直接使っています。これは意図的な例外です。

### Socket.io で通知されるイベント

学習中、サーバーからブラウザへ以下のイベントが流れてきます。

| イベント | 内容 |
|---|---|
| `train:progress` | `{ epoch, loss }` または `{ status: "converged", ... }` |
| `train:done` | `{ success: true/false }` |

---

## フロントエンド（App.vue）

### 画面の構成

- 上部：セグメント選択ボタン（ステータスドット付き）
- 中部：試聴プレイヤー + ラベル入力フォーム
- 下部：学習パネル（Alpha・Train ボタン・ログ）

### 主要な state

```js
config          // config.json の内容。フォームの形を決める
segments        // セグメント ID の配列
labelset        // 保存済みラベルの全データ
trainedIds      // 最後の学習に含まれたセグメントの ID 一覧
currentSegment  // 今選択中のセグメント
currentLabels   // フォームに表示中の値（UI 表現）
```

### セグメントのステータス

各ボタンに表示される色ドットは次のように決まります。

```js
const segmentStatus = (id) => {
  if (!labelset.value.data[id]) return 'unlabeled' // グレー：未ラベル
  if (trainedIds.value.includes(id)) return 'trained' // 緑：学習済み
  return 'pending' // 黄：ラベルあり・未学習
}
```

---

## ラベルのシリアライズ（最重要）

### なぜシリアライズが必要か

`labelset.json` は Python（Module D）と共有するファイルです。
Python が扱いやすいよう、値はすべて **0.0〜1.0 のフラット配列** で保存します。

でも UI では色をカラーピッカーで選んだり、ドロップダウンで選択したりしますよね。
この「UI の値 ↔ フラット配列」の変換を担うのが `encodeLabels` と `decodeLabels` です。

```
UI の値                    保存される値
─────────────────────────────────────────────────
"#ff8800"              →  [1.0, 0.533, 0.0]      (R/255, G/255, B/255)
dropdown: index 1      →  [0.5]                  (index / (選択肢数 - 1))
slider: 0.7            →  [0.7]                  (そのまま)
checkboxes: [✓,✗,✓]   →  [1.0, 0.0, 1.0]        (true=1.0, false=0.0)
```

### 新しいフィールドタイプを追加するとき

この2関数（`encodeLabels` と `decodeLabels`）は必ずペアで更新してください。片方だけ変更すると保存と読み込みで値がずれます。

以下のテンプレートをコピーして、`YOUR_TYPE` と各コメント部分を書き換えてください。

#### 1. `data/config.json` にフィールド定義を追加

```json
{
  "id": "YOUR_ID",
  "type": "YOUR_TYPE",
  "label": "フォームに表示するラベル名"
  // 必要なオプションをここに追加（例: "options", "min", "max" など）
}
```

#### 2. `App.vue` テンプレートに入力 UI を追加

`v-if` が並んでいるブロックの末尾に追加します。

```html
<div v-if="field.type === 'YOUR_TYPE'">
  <!-- ユーザーが操作する入力 UI をここに書く -->
  <!-- v-model="currentLabels[index]" でフォームの値と紐付ける -->
</div>
```

#### 3. `encodeLabels` に変換ロジックを追加（UI 値 → 0.0〜1.0）

`encodeLabels` 関数内の `} else {` の直前に追加します。

```js
} else if (field.type === 'YOUR_TYPE') {
  // val に currentLabels[index] の値が入っている
  // 0.0〜1.0 に変換して flat.push() する
  flat.push(/* val を 0.0〜1.0 に変換する式 */)
}
```

#### 4. `decodeLabels` に逆変換ロジックを追加（0.0〜1.0 → UI 値）

`decodeLabels` 関数内の `} else {` の直前に追加します。

```js
} else if (field.type === 'YOUR_TYPE') {
  // flat[idx++] に 0.0〜1.0 の値が入っている
  // UI で表示できる値に変換して return する
  return /* flat[idx++] を UI 値に変換する式 */
}
```

#### 5. `initLabels` にデフォルト値を追加

`initLabels` 関数内の `return null` の直前に追加します。

```js
if (field.type === 'YOUR_TYPE') return /* 初期値 */
```

---

#### 実装例：5段階評価 `rating` を追加する

上のテンプレートを埋めると次のようになります。

```json
// config.json
{ "id": "quality", "type": "rating", "max": 5, "label": "クオリティ" }
```

```html
<!-- App.vue テンプレート -->
<div v-if="field.type === 'rating'" class="flex gap-2">
  <button v-for="n in field.max" :key="n"
    @click="currentLabels[index] = n"
    :class="n <= currentLabels[index] ? 'text-yellow-400' : 'text-gray-300'"
  >★</button>
</div>
```

```js
// encodeLabels： 1〜5 を 0.0〜1.0 に変換
} else if (field.type === 'rating') {
  flat.push((val - 1) / (field.max - 1))
}

// decodeLabels： 0.0〜1.0 を 1〜5 に戻す
} else if (field.type === 'rating') {
  return Math.round(flat[idx++] * (field.max - 1)) + 1
}

// initLabels： デフォルトは中間値
if (field.type === 'rating') return Math.ceil(field.max / 2)
```

---

## 2つの主要フロー

### AI Suggestion（未ラベルセグメントを選んだとき）

```
セグメント選択（ラベルなし）
  → predict(silent=true) が自動実行
  → POST /api/predict
  → predict.py が全セグメントを推論
  → 該当 ID の結果を decodeLabels で UI 値に変換
  → フォームにプレビュー表示（まだ保存はされていない）
  → ユーザーが確認・修正 → Save Labels
```

モデルがまだない場合はプレビューをスキップするだけです（エラーにはなりません）。

### 学習（Train Model ボタンを押したとき）

```
POST /api/train → すぐに { success: true } が返る（非同期）
  ↓ 裏でプロセスが動く
  train:progress { epoch, loss } がリアルタイムで届く
  train:progress { status: "converged" } で収束を通知
  train:done { success: true } でプロセス終了
  → fetchTrainMeta() を呼び出してステータスドットを更新
```

---

## コードを拡張するときの指針

- state の追加は `ref()` + `const` で
- API 呼び出しは axios を使う（fetch は使わない）
- ローディング中の処理は `try/finally` で `isLoading` を必ず解除する
- `App.vue` が大きくなってきたら以下の単位でコンポーネントに分割できます

| 分割候補 | 責務 |
|---|---|
| `SegmentList.vue` | セグメント一覧とステータスドット |
| `LabelForm.vue` | 動的フォームと Save/AI Suggestion |
| `TrainingPanel.vue` | 学習操作パネル |
| `FieldRenderer.vue` | フィールドタイプごとの入力 UI |

分割する際は `encodeLabels` / `decodeLabels` を `src/utils/labels.js` に切り出すと
どのコンポーネントからも使いやすくなります。
