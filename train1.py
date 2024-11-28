import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

# 数据预处理
def preprocess_data(input_texts, target_texts, num_samples=10000):
    """
    对输入数据和目标数据进行预处理，并构建词汇表和字符到索引的映射。
    """
    input_characters = set()
    target_characters = set()
    for input_text, target_text in zip(input_texts, target_texts):
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))

    input_token_index = {char: i for i, char in enumerate(input_characters)}
    target_token_index = {char: i for i, char in enumerate(target_characters)}

    max_encoder_seq_length = max([len(text) for text in input_texts])
    max_decoder_seq_length = max([len(text) for text in target_texts])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, len(input_characters)),
        dtype="float32",
    )
    decoder_input_data = np.zeros(
        (len(target_texts), max_decoder_seq_length, len(target_characters)),
        dtype="float32",
    )
    decoder_target_data = np.zeros(
        (len(target_texts), max_decoder_seq_length, len(target_characters)),
        dtype="float32",
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

    return (encoder_input_data, decoder_input_data, decoder_target_data,
            input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length)

# 构建模型
def build_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    """
    构建编码器-解码器模型。
    """
    # 编码器
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # 我们将 encoder_states 传递给解码器
    encoder_states = [state_h, state_c]

    # 解码器
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # 定义模型
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense

# 构建推理模型
def build_inference_model(encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense, latent_dim):
    """
    构建推理模型，用于编码和解码阶段的推断。
    """
    # 编码器模型
    encoder_model = Model(encoder_inputs, encoder_states)

    # 解码器模型
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model

# 解码序列
def decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, len(target_token_index)))
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        print(f"Step {len(decoded_sentence)}: {sampled_char}")  # 调试信息

        if (sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, len(target_token_index)))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]

    return decoded_sentence

# 主程序
if __name__ == "__main__":
    # 示例数据（中文句子）
    input_texts = ["人之初 性本善", "性相近 习相远", "苟不教 性乃迁"]
    target_texts = ["\t人之初 性本善\n", "\t性相近 习相远\n","\t苟不教 性乃迁\n"]

    # 数据预处理
    (encoder_input_data, decoder_input_data, decoder_target_data,
     input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length) = preprocess_data(input_texts, target_texts)

    # 反向字符索引
    reverse_input_char_index = {i: char for char, i in input_token_index.items()}
    reverse_target_char_index = {i: char for char, i in target_token_index.items()}

    # 模型参数
    num_encoder_tokens = len(input_token_index)
    num_decoder_tokens = len(target_token_index)
    latent_dim = 256  # LSTM 隐藏层维度

    # 构建模型
    model, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense = build_model(
        num_encoder_tokens, num_decoder_tokens, latent_dim
    )

    # 编译并训练模型
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=64,
        epochs=500,  # 增加到 500
        validation_split=0,
    )

    # 保存模型
    model.save("s2s.h5")

    # 构建推理模型
    encoder_model, decoder_model = build_inference_model(
        encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense, latent_dim
    )

    # 测试解码
    for seq_index in range(len(input_texts)):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index)
        print("-")
        print("输入句子:", input_texts[seq_index])
        print("解码句子:", decoded_sentence)
