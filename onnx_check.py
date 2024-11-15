import numpy as np
import onnxruntime as ort

# モデルの読み込み
session = ort.InferenceSession('your_model.onnx')

# モデルの入力の名前と形状を取得
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f'Input name: {input_name}, Input shape: {input_shape}')

# 入力データの作成（例として、画像データを仮定）
input_data = np.random.randn(*input_shape).astype(np.float32)

# 推論の実行
result = session.run(None, {input_name: input_data})
print(result)
