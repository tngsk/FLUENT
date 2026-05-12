import dotenv from "dotenv";
import { fileURLToPath } from "url";
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.join(__dirname, "../.env") });

import express from "express";
import cors from "cors";
import fs from "fs/promises";
import { constants } from "fs";
import { createServer } from "http";
import { Server } from "socket.io";
import { spawn } from "child_process";

const DATA_DIR = path.resolve(__dirname, "..", process.env.DATA_DIR || "data");
const PYTHON_BIN = path.resolve(
  __dirname,
  "..",
  process.env.PYTHON_PATH || ".venv/bin/python",
);
const ROOT_DIR = path.resolve(__dirname, "..");
const PORT = process.env.PORT || 3000;

const app = express();
const httpServer = createServer(app);
const allowedOrigins = (
  process.env.ALLOWED_ORIGINS || "http://localhost:5173"
).split(",");

const io = new Server(httpServer, { cors: { origin: allowedOrigins } });

app.use(
  cors({
    origin: (origin, cb) =>
      !origin || allowedOrigins.includes(origin)
        ? cb(null, true)
        : cb(new Error("Not allowed")),
  }),
);
app.use(express.json());
app.use("/data", express.static(DATA_DIR));

app.get("/api/config", async (req, res) => {
  try {
    res.json(
      JSON.parse(await fs.readFile(path.join(DATA_DIR, "config.json"), "utf8")),
    );
  } catch {
    res.status(500).json({ error: "Failed to read config" });
  }
});

app.get("/api/segments", async (req, res) => {
  try {
    const files = await fs.readdir(path.join(DATA_DIR, "segments"));
    res.json(
      files
        .filter((f) => f.endsWith(".wav"))
        .map((f) => f.replace(".wav", ""))
        .sort(),
    );
  } catch {
    res.status(500).json({ error: "Failed to read segments" });
  }
});

app.get("/api/labels", async (req, res) => {
  const labelsetPath = path.join(DATA_DIR, "labelset.json");
  try {
    await fs.access(labelsetPath, constants.F_OK);
    res.json(JSON.parse(await fs.readFile(labelsetPath, "utf8")));
  } catch {
    res.json({ cols: 0, data: {} });
  }
});

app.post("/api/labels", async (req, res) => {
  const { id, labels, cols } = req.body;
  const labelsetPath = path.join(DATA_DIR, "labelset.json");
  let labelset = { cols: cols || 0, data: {} };
  try {
    await fs.access(labelsetPath, constants.F_OK);
    const data = await fs.readFile(labelsetPath, "utf8");
    labelset = JSON.parse(data);
    const backupDir = path.join(DATA_DIR, "backups");
    await fs.mkdir(backupDir, { recursive: true });
    await fs.writeFile(
      path.join(
        backupDir,
        `labelset-${new Date().toISOString().replace(/[:.]/g, "-")}.json`,
      ),
      data,
    );
  } catch {}
  labelset.cols = cols || labelset.cols;
  labelset.data[id] = labels;
  try {
    await fs.writeFile(labelsetPath, JSON.stringify(labelset, null, 2));
    res.json({ success: true });
  } catch {
    res.status(500).json({ error: "Failed to save labels" });
  }
});

app.post("/api/predict", (req, res) => {
  const { id } = req.body;
  const proc = spawn(PYTHON_BIN, [path.join(ROOT_DIR, "python/predict.py")], {
    cwd: ROOT_DIR,
  });
  let out = "",
    err = "";
  proc.stdout.on("data", (d) => (out += d));
  proc.stderr.on("data", (d) => (err += d));
  proc.on("close", (code) => {
    if (code !== 0)
      return res.status(500).json({ error: "Prediction failed", details: err });
    try {
      const predictions = JSON.parse(out);
      predictions.data?.[id]
        ? res.json({ success: true, prediction: predictions.data[id] })
        : res.json({ success: false, message: "No prediction found" });
    } catch {
      res.status(500).json({ error: "Invalid JSON", details: err });
    }
  });
});

let trainingProcess = null;
app.post("/api/train", (req, res) => {
  if (trainingProcess)
    return res.status(409).json({ error: "Training already in progress" });
  const alpha = parseFloat(req.body.alpha) || 0.01;
  const proc = spawn(
    PYTHON_BIN,
    [path.join(ROOT_DIR, "python/train.py"), "--alpha", String(alpha)],
    { cwd: ROOT_DIR },
  );
  trainingProcess = proc;
  res.json({ success: true });
  proc.stdout.on("data", (data) => {
    data
      .toString()
      .trim()
      .split("\n")
      .filter(Boolean)
      .forEach((line) => {
        try {
          io.emit("train:progress", JSON.parse(line));
        } catch {}
      });
  });
  proc.stderr.on("data", (data) =>
    console.error("[train.py]", data.toString().trim()),
  );
  proc.on("close", (code) => {
    trainingProcess = null;
    io.emit("train:done", { success: code === 0 });
  });
});

httpServer.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
