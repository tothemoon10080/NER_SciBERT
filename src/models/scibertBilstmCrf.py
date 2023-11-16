from keras.layers import Input, Bidirectional, LSTM, Dense, TimeDistributed
from keras.models import Model
from tf2crf import CRF, ModelWithCRFLoss
from transformers import AutoModel, TFAutoModel
import os

def create_model(max_len, label_to_id, model_path):
    # 检查模型路径是否提供且正确
    if not model_path or not os.path.exists(model_path):
        print("Invalid model path provided. Please provide a valid model path.")
        return None

    # 检查模型文件夹中是否存在 TensorFlow 模型文件
    if os.path.exists(os.path.join(model_path, "tf_model.h5")):
        try:
            # 加载 TensorFlow 模型
            scibert_model = TFAutoModel.from_pretrained(model_path, from_pt=False)
        except Exception as e:
            print(f"An error occurred while loading the TensorFlow model: {e}")
            return None
    else:
        print("Unsupported model file type. Please provide a TensorFlow model.")
        return None


    # 定义模型的输入层
    input_ids_layer = Input(shape=(max_len,), dtype='int32', name="input_ids")
    attention_masks_layer = Input(shape=(max_len,), dtype='int32', name="attention_masks")

    # 通过模型获取嵌入层输出
    embeddings = scibert_model(input_ids_layer, attention_mask=attention_masks_layer).last_hidden_state

    # 添加BiLSTM层
    bilstm = Bidirectional(LSTM(units=128, return_sequences=True))(embeddings)

    # 添加TimeDistributed层
    dense = TimeDistributed(Dense(50, activation="relu"))(bilstm)

    # 添加CRF层
    crf = CRF(len(label_to_id), name="crf_layer")
    output = crf(dense)

    # 构建模型
    model = Model(inputs=[input_ids_layer, attention_masks_layer], outputs=output)
    model_with_crf = ModelWithCRFLoss(model)

    return model_with_crf

