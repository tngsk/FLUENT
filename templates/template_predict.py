import json
import numpy as np
import pickle
import sys

def simple_predict(features_json, model_path, output_json=None):
    """
    学習済みモデルと特徴量データを読み込み、推論を行う最小構成のプレディクター。
    出力は 0.0 ~ 1.0 にクリップされます。
    """
    # 1. モデルと特徴量の読み込み
    try:
        with open(model_path, 'rb') as f:
            mlp = pickle.load(f)
        with open(features_json, 'r') as f:
            features_data = json.load(f)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    file_ids = []
    features_list = []

    # 2. データの準備
    for file_id, feature_vector in features_data.get("data", {}).items():
        file_ids.append(file_id)
        features_list.append(feature_vector)

    if not features_list:
        print(json.dumps({"error": "No features found to predict."}))
        return

    X = np.array(features_list)

    # 3. 推論の実行
    y_pred = mlp.predict(X)

    # 4. 出力のクリップ (0.0 ~ 1.0)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    # 5. 出力データの整形
    predictions = {}
    for i, file_id in enumerate(file_ids):
        # numpy配列をリストに変換して格納
        predictions[file_id] = y_pred[i].tolist() if isinstance(y_pred[i], np.ndarray) else [float(y_pred[i])]

    output_data = {
        "cols": len(predictions[file_ids[0]]),
        "data": predictions
    }

    # 6. 結果の出力
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved predictions to {output_json}")
    else:
        # JSON文字列として標準出力へ (Node.js等が受け取るため)
        print(json.dumps(output_data))

if __name__ == "__main__":
    # 使用例
    # simple_predict("features.json", "model.pkl", output_json="predictions.json")
    print("This is a template for predicting labels.")
