import json
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def simple_trainer(features_json, labels_json, model_output="model.pkl"):
    """
    特徴量とラベルのJSONを読み込み、MLPRegressorで学習を行う最小構成のトレイナー。
    """
    # 1. データの読み込み
    try:
        with open(features_json, 'r') as f:
            features_data = json.load(f)
        with open(labels_json, 'r') as f:
            labels_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}. Create dummy data or run extractor first.")
        return

    X_list = []
    y_list = []

    # 2. データの紐付け
    # subjectId を "user1" と仮定（プロジェクト構成に依存）
    subject_id = "user1"

    for file_id, feature_vector in features_data.get("data", {}).items():
        if file_id in labels_data.get("data", {}):
            label_entry = labels_data["data"][file_id]
            # labels_data が {"example-001": {"user1": [0.5, 1.0]}} のような構造の場合
            if subject_id in label_entry:
                y_list.append(label_entry[subject_id])
                X_list.append(feature_vector)
            # labels_data が直接 {"example-001": [0.5, 1.0]} の場合
            elif isinstance(label_entry, list):
                 y_list.append(label_entry)
                 X_list.append(feature_vector)

    if not X_list:
        print("No matching training data found.")
        return

    X = np.array(X_list)
    y = np.array(y_list)

    # 3. 学習用とテスト用に分割（オプション）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. モデルの構築と学習
    print("Training model...")
    # alphaはL2正則化項。max_iterは最大エポック数。
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, alpha=0.01, random_state=42)
    mlp.fit(X_train, y_train)

    # 簡単な評価
    score = mlp.score(X_test, y_test)
    print(f"Test R^2 Score: {score:.4f}")

    # 5. モデルの保存
    with open(model_output, 'wb') as f:
        pickle.dump(mlp, f)
    print(f"Model saved to {model_output}")

if __name__ == "__main__":
    # 使用例
    # simple_trainer("features.json", "labels.json", "model.pkl")
    print("This is a template for training a model.")

    # --- ダミーデータの生成例 (テスト用) ---
    # import os
    # if not os.path.exists("features.json"):
    #     dummy_f = {"cols": 2, "data": {"seg1": [0.1, 0.2], "seg2": [0.8, 0.9], "seg3": [0.5, 0.5]}}
    #     dummy_l = {"cols": 1, "data": {"seg1": {"user1": [0.0]}, "seg2": {"user1": [1.0]}, "seg3": {"user1": [0.5]}}}
    #     json.dump(dummy_f, open("features.json", "w"))
    #     json.dump(dummy_l, open("labels.json", "w"))
    #     print("Created dummy data. Run again to train.")
