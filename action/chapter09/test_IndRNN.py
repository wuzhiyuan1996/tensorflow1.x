
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import jieba
import tensorflow.contrib as contrib


def gen_data(path):
    START_TOKEN = 0
    END_TOKEN = 1
    alltext = []
    for file in os.listdir(path):
        with open(path+file, 'r', encoding='utf-8') as f:
            strtext = f.read().split('\n')
            strtext = list(filter(lambda x:len(x)>0, strtext)) # 过滤空字符串
            strtext = list(map(lambda x:" ".join(jieba.cut(x.replace('-', '').replace(' ', '')))))
            print(file, strtext[:2])
            alltext = alltext + strtext
            print(len(alltext))

    top_k = 5000 # 过滤文本，选出5000个
    #  oov_token: if given, it will be added to word_index and used to
    #  replace out-of-vocabulary words during text_to_sequence calls
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>")
    # Updates internal vocabulary based on a list of texts.
    tokenizer.fit_on_texts(texts=alltext)
    # word_index 保存所有word对应的编号
    # 出现的频率越大，排在越前面，因此是<=
    tokenizer.word_index = {key:value for key, value in tokenizer.word_index.items() if value <= top_k}
    tokenizer.word_index[tokenizer.oov_token] = top_k + 1 # oov_token 排最后

    tokenizer.word_index['<start>'] = START_TOKEN # 标记输入样本的起始位置
    tokenizer.word_index['<end>'] = END_TOKEN # 标记输入样本的结束位置

    # 反向字典
    indexed_word = {value:key for key, value in tokenizer.word_index.items()}
    print(len(indexed_word))


def test(tokenizer, alltext):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>")
    # Transforms each text in texts to a sequence of integers
    train_seqs = tokenizer.texts_to_sequences(texts=alltext)

    input_seq, output_seq = train_seqs[0::2], train_seqs[1::2] # 拆分成问题和答案
    print(len(input_seq), len(output_seq))
    # 将所有序列补充到相同的长度，padding的方式为向后填充，填充的值为value
    input_vectors = tf.keras.preprocessing.sequence.pad_sequences(sequences=input_seq, padding='post', value=END_TOKEN)
    output_vectors = tf.keras.preprocessing.sequence.pad_sequences(sequences=output_seq, padding='post', value=END_TOKEN)
    # 为所有句子添加结束标志
    end = np.ones_like(input_vectors[:, 0]) # input_vectors:(a,b) -> end (a,)
    end = np.reshape(a=end, newshape=[-1, 1])
    input_vectors = np.concatenate([input_vectors, end], axis=1)
    output_vectors = np.concatenate([output_vectors, end], axis=1)
    in_max_length = len(input_vectors[0])
    out_max_length = len(output_vectors[0])

    input_vectors_train, input_vectors_val, output_vectors_train, output_vectors_val = train_test_split(input_vectors, output_vectors, test_size=0.2, random_state=0)


def seq2seq(mode, features, labels, params):
    vocab_size = params['vocab_size']
    embed_dim = params['embed_dim']
    num_units = params['num_units']
    output_max_length = params['output_max_length']
    print("特征和标签：", features.name, labels.name)
    batch_size = tf.shape(features)[0]
    # 重复，multiples: 1-D. Length must be the same as the number of dimensions in `input`
    start_tokens = tf.tile(inputs=[START_TOKEN], multiples=[batch_size])
    train_output = tf.concat([tf.expand_dims(input=start_tokens, axis=1), labels], axis=1)
    # Returns the truth value of (x != y) element-wise.
    # 输入的数据中，每一个序列的实际长度。shape：[batch_size, 1]
    input_lengths = tf.reduce_sum(tf.cast(tf.not_equal(x=features, y=END_TOKEN), dtype=tf.int32), axis=1, name="input_lens")
    output_lengths = tf.reduce_sum(tf.cast(tf.not_equal(x=train_output, y=END_TOKEN), dtype=tf.int32), axis=1, name="output_lens")
    tf.keras.layers.Embedding
    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable(name='embeddings')
    # Independently Recurrent Neural Network (IndRNN) cell
    # DeviceWrapper: Operator that ensures an RNNCell runs on a particular device.
    # DeviceWrapper: 是否类似于with tf.device("/device:GPU:0"):
    IndRNN_Cell = tf.nn.rnn_cell.DeviceWrapper(contrib.rnn.IndRNNCell(num_units=num_units), "/device:GPU:0")
    IndyLSTM_Cell = tf.nn.rnn_cell.DeviceWrapper(contrib.rnn.IndyLSTMCell(num_units=num_units), "/device:GPU:0")
    # Create a RNN cell composed sequentially of a number of RNNCells.
    multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[IndRNN_Cell, IndyLSTM_Cell], state_is_tuple=True)
    # dynamic_rnn： 动态的RNN，不需要指定输入序列的长度，既不需要指定时间步
    # sequence_length: 每个序列的真实长度， cell 是一个list，len(cell)表示RNN的层数
    # encoder_outputs:每一个时间步的输出，shape: [batch_size, max_length, num_units]
    # encoder_final_state:
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        cell=multi_cell, inputs=input_embed, sequence_length=input_lengths, dtype=tf.float32)
    if useScheduled:
        # contrib.seq2seq.ScheduledOutputTrainingHelper， 直接对输出进行计划采样（p的概率选择输出概率最大的词，1-p的概率选择真实的词）
        # 对输出的广义伯努利分布（属于多项式分布）进行计划采样
        # output_dim是真实值（target）
        train_helper = contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=output_embeded, sequence_length=tf.tile([output_max_length], [batch_size]), sampling_probability=0.3)
    else:
        # 用于训练过程，将上一时间步的真实值作为下一时间步的输入
        train_helper = contrib.seq2seq.TrainingHelper(
            inputs=output_embeded, sequence_length=tf.tile(input=[output_max_length], multiples=[batch_size]))
    # 用在模型的预测(使用)过程中，从输出结果中找到概率最大的embedding，转换为词
    #  embedding: A callable that takes a vector tensor of `ids` (argmax ids),
    #  or the `params` argument for `embedding_lookup`. The returned tensor will be passed to the decoder input.
    pred_helper = contrib.seq2seq.GreedyEmbeddingHelper(
        embeddings=embeddings, start_tokens=tf.tile([START_TOKENS], [batch_size]), end_token=END_TOKEN)

    # contrib.seq2seq.SampleEmbeddingHelper 继承自GreedyEmbeddingHelper，从生产的概率分布中采样
    def decode(helper, scope, reuse=None):
        with tf.variable_scope(name_or_scope=scope, reuse=reuse):
            attention_mechanism = contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=encoder_outputs, memory_sequence_length=input_lengths)
            cell = contrib.rnn.IndRNNCell(num_units=num_units)
            if reuse == None:
                keep_prob = 0.8
            else:
                keep_prob = 1.0
            # attention_layer_size, attention的输出大小
            attn_cell = contrib.seq2seq.AttentionWrapper(
                cell=cell, attention_mechanism=attention_mechanism, attention_layer_size=num_units/2)
            # Operator adding an output projection to the given cell.
            out_cell = contrib.rnn.OutputProjectionWrapper(
                cell=attn_cell, output_size=vocab_size, reuse=reuse)
            decoder = contrib.seq2seq.BasicDecoder(
                cell=out_cell, helper=helper, initial_state=out_cell.zero_state(batch_size=batch_size, dtype=tf.float32))
            # impute_finished – Python boolean. If `True`, then states for batch entries which are marked as
            # finished get copied through and the corresponding outputs get zeroed out. This causes some slowdown
            # at each time step, but ensures that the final state and outputs have the correct values and that backprop
            # ignores time steps that were marked as finished.
            # 输出为`(final_outputs, final_state, final_sequence_lengths)`.
            outputs = contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False, impute_finished=True, maximum_iterations=output_max_length)
            return outputs[0]

    train_outputs = decode(train_helper, 'decode')
    pred_outputs = decode(pred_helper, 'decode', reuse=True)

    masks = tf.sequence_mask(lengths=output_lengths, maxlen=output_max_length, dtype=tf.float32, name='masks')
    # logits – A Tensor of shape `[batch_size, sequence_length, num_decoder_symbols]`
    # targets – A Tensor of shape `[batch_size, sequence_length]` and dtype int. The target represents the true class at each timestep.
    # weights – A Tensor of shape `[batch_size, sequence_length]` and dtype float.
    # `weights` constitutes the weighting of each prediction in the sequence.
    # When using `weights` as masking, set all valid timesteps to 1 and all padded timesteps to 0,
    loss = contrib.seq2seq.sequence_loss(logits=train_outputs.rnn_output, targets=labels, weights=masks)
    train_op = contrib.layers.optimize_loss(
        loss=loss, global_step=tf.train.get_global_step(), optimizer=params.get('optimizer', 'Adam'),
        learning_rate=params.get('learning_rate', 0.001), summaries=['loss', 'learning_rate']
    )
    # Creates a validated EstimatorSpec instance.
    # Depending on the value of mode, different arguments are required. Namely
    # For mode == ModeKeys.TRAIN: required fields are loss and train_op.
    # For mode == ModeKeys.EVAL: required field is loss.
    # For mode == ModeKeys.PREDICT: required fields are predictions
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=pred_outputs.sample_id, loss=loss, train_op=train_op
    )

