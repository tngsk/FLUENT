import express from 'express';
import cors from 'cors';
import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// ミドルウェアの設定
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// --- モックデータの準備 ---
const MOCK_DATA_DIR = path.join(__dirname, 'data');

// サーバー起動時にモックデータディレクトリを作成
async function setupMockData() {
    try {
        await fs.mkdir(MOCK_DATA_DIR, { recursive: true });
        const dummyLabels = { cols: 1, data: { "seg1": { "user1": [1.0] } } };
        await fs.writeFile(path.join(MOCK_DATA_DIR, 'labelset.json'), JSON.stringify(dummyLabels), 'utf8').catch(() => {});
    } catch (e) {
        console.error("Setup error:", e);
    }
}
setupMockData();

// --- API エンドポイント ---

// 1. ラベルデータの取得 (GET)
app.get('/api/labels', async (req, res) => {
    try {
        const data = await fs.readFile(path.join(MOCK_DATA_DIR, 'labelset.json'), 'utf8');
        res.json(JSON.parse(data));
    } catch (e) {
        res.status(500).json({ error: "Failed to read labels" });
    }
});

// 2. ラベルデータの保存 (POST)
app.post('/api/labels', async (req, res) => {
    try {
        // 実際のアプリケーションではここでデータをマージしたりバリデーションを行います
        await fs.writeFile(path.join(MOCK_DATA_DIR, 'labelset.json'), JSON.stringify(req.body, null, 2));
        res.json({ success: true });
    } catch (e) {
        res.status(500).json({ error: "Failed to save labels" });
    }
});

// 3. Pythonスクリプトの実行（推論の例）
app.post('/api/predict', async (req, res) => {
    // 実際には req.body.id などを受け取って引数に渡します
    const pythonProcess = spawn('python', [path.join(__dirname, 'template_predict.py')]); // パスは適宜調整してください

    let output = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        error += data.toString();
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            try {
                // stdoutからJSONをパースして返す
                const result = JSON.parse(output);
                res.json({ success: true, result });
            } catch (e) {
                res.json({ success: true, message: "Raw output", raw: output });
            }
        } else {
            res.status(500).json({ success: false, error });
        }
    });
});

// サーバー起動
app.listen(PORT, () => {
    console.log(`Template server listening on port ${PORT}`);
});
