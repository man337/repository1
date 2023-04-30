'''
attention-lstm预测二维数据
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# RNN模型相关参数
HIDDEN_SIZE = 128                           # LSTM中隐藏节点的个数,定义输出及状态的向量维度。
TIME_STEPS = 10                             # 神经网络的训练序列长度
BATCH_SIZE = 3                              # batch大小。
epochs = 15                                  # 训练轮数。(50效果还行)
plt.rcParams['font.sans-serif'] = ['FangSong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
if_global = False

sc1 = MinMaxScaler(feature_range=(0, 1))


def fun_data3():
    init_df = pd.read_csv(
                'data/unique_lane_data3.csv',
                header=0,
                usecols=[
                           "Vehicle_ID",
                           "Global_Time",
                           "Global_X",
                           "Global_Y",
                           "Local_X",
                           "Local_Y",
                           "v_Vel",
                           # "Lane_ID",
                         ],
                )
    # print(init_df.shape[0])   # 2888
    sep_row = init_df[init_df['Vehicle_ID'] == 'Vehicle_ID'].index
    sep_num = []
    data0 = []
    time0 = []
    sep_num.append(0)
    vehicle_seq = []
    time_seq = []
    last_vehicle_ID = init_df.loc[0, 'Vehicle_ID']
    # 处理数据。shape：轨迹总条数（125）*轨迹长度（不定）*特征维数（2）
    for i in range(0, init_df.shape[0]):
        if i in sep_row:
            sep_num.append(len(data0) + 1)  # 储存的是下一batch的第一个轨迹的索引（从0开始）。
            # print(init_df[i:i+1])
        else:
            now_vehicle_ID = init_df.loc[i, 'Vehicle_ID']
            if if_global:
                one_point = [float(init_df.loc[i, 'Global_X']), float(init_df.loc[i, 'Global_Y'])]
            else:
                one_point = [float(init_df.loc[i, 'Local_X']), float(init_df.loc[i, 'Local_Y'])]
            one_time = int(init_df.loc[i, 'Global_Time'])
            if last_vehicle_ID == now_vehicle_ID:
                vehicle_seq.append(one_point)
                time_seq.append(one_time)
            else:
                data0.append(vehicle_seq)
                time0.append(time_seq)
                vehicle_seq = []
                time_seq = []
                vehicle_seq.append(one_point)
                time_seq.append(one_time)
            if i == init_df.shape[0] - 1:
                data0.append(vehicle_seq)
                time0.append(time_seq)
            last_vehicle_ID = now_vehicle_ID
    # print(sep_num)
    # 处理数据。每段路只取3条轨迹（因此后面lstm的batchsize=3）。0,3,6……
    data1 = []
    time1 = []
    min_t = []
    for i in sep_num:
        data1.append(data0[i])
        data1.append(data0[i+1])
        data1.append(data0[i+2])
        temp_min = []
        time1.append(time0[i])
        temp_min.append(min(time0[i]))
        time1.append(time0[i+1])
        temp_min.append(min(time0[i]))
        time1.append(time0[i+2])
        temp_min.append(min(time0[i]))
        min_t.append(min(temp_min))
    # 共20（15）秒。分40（30）段
    mid_result = []
    for i in range(0, len(data1)):
        start_t = min_t[int(i / 3)]
        # t_tidy = np.arange(0, 150, 5)  # 15秒，30个数据
        t_tidy = np.arange(0, 200, 5)  # 20秒，40个数据
        # t_tidy = np.arange(0, 250, 5)   # 25秒，50个数据
        init_seq = np.array(data1[i])
        if if_global:
            init_seq0 = init_seq[:, 0] - 1966700
            init_seq1 = init_seq[:, 1] - 570500
        else:
            init_seq0 = init_seq[:, 0]
            init_seq1 = init_seq[:, 1] - 550
        t_seq = np.array(time1[i]) - start_t
        z0 = np.polyfit(t_seq, init_seq0, 6)  # 横向变化很小(这组6，4效果还行)
        z1 = np.polyfit(t_seq, init_seq1, 4)  # 纵向变化很大
        out0 = np.polyval(z0, t_tidy)
        out0 = np.expand_dims(out0, axis=1)
        out1 = np.polyval(z1, t_tidy)
        out1 = np.expand_dims(out1, axis=1)
        mid_result.append(np.concatenate([out0, out1], axis=1))

        # 显示双坐标拟合效果
    #     plt.figure(figsize=(8, 4))
    #     print(i)
    #     plt.plot(init_seq0, init_seq1, 'bo', label='原数据', linestyle='solid')  # 蓝色虚线
    #     plt.plot(out0, out1, 'r.', label='拟合', linestyle='solid')  # 红色点状
    #     plt.xlabel('x(m)')
    #     plt.ylabel('y(m)')
    #     plt.legend(loc=0)
    #     # if i % 3 == 2:
    #     plt.show()
    # input()

    #     # 显示单坐标拟合效果
    #     plt.figure(figsize=(8, 4))
    #     # plt.plot(t_seq, init_seq0, 'bo', label='原数据')  # 蓝色虚线
    #     # plt.plot(t_tidy, out0, 'r.', label='拟合')  # 红色点状
    #     plt.plot(t_seq, init_seq1, 'bo', label='原数据')  # 蓝色虚线
    #     plt.plot(t_tidy, out1, 'r.', label='拟合')  # 红色点状
    #     plt.xlabel('t(0.1s)')
    #     plt.ylabel('函数值(m)')
    #     plt.legend(loc=0)
    #     plt.show()
    # input()

    # 60*40*2
    mid_result = np.array(mid_result)
    # 前1、后n个值很离谱，得去掉
    len1 = mid_result.shape[1]
    mid_result = mid_result[:, 1:len1 - 5, :]
    # 打乱
    # np.random.shuffle(mid_result)

    # 归一化 60*34*2。非常有用
    mid_shape = mid_result.shape
    mid_result = np.reshape(mid_result, (mid_shape[0] * mid_shape[1], mid_shape[2]))
    mid_result = sc1.fit_transform(mid_result)
    mid_result = np.reshape(mid_result, (mid_shape[0], mid_shape[1], mid_shape[2]))

    # # 查看数据集效果
    # plt.figure()
    # for i in range(0, mid_result.shape[0]):
    #     if i % 3 == 0:
    #         plt.plot(mid_result[i, :, 0], mid_result[i, :, 1], 'r.', label='车辆1', linestyle='solid', color='red')
    #     if i % 3 == 1:
    #         plt.plot(mid_result[i, :, 0], mid_result[i, :, 1], 'r.', label='车辆2', linestyle='solid', color='green')
    #     if i % 3 == 2:
    #         plt.plot(mid_result[i, :, 0], mid_result[i, :, 1], 'r.', label='车辆3', linestyle='solid', color='blue')
    #     plt.xlabel('x(m)')
    #     plt.ylabel('y(m)')
    #     plt.legend()
    #     print(i)
    #     if i % 3 == 2:
    #         # print(int(i / 3) + 1)
    #         plt.show()
    # input()

    # 40个值，25个数据。（30，15）

    TIME_STEPS = 10
    train_per = 0.8
    num_data = mid_result.shape[0]  # 60
    total_timestep = mid_result.shape[1]  # 36?
    # print(total_timestep)
    # 12*10*2
    test_x = mid_result[int(train_per * num_data):, total_timestep - 2 * TIME_STEPS:total_timestep - TIME_STEPS, :]
    test_y = mid_result[int(train_per * num_data):, total_timestep - TIME_STEPS:total_timestep, :]
    X, Y = [], []
    for i in range(mid_result.shape[1] - TIME_STEPS):
        X.append(mid_result[:, i: i + TIME_STEPS, :])  # 用[0]至[9]个特征
        Y.append(mid_result[:, i + TIME_STEPS, :])  # 预测[10]这个值
    X = np.array(X, dtype=np.float32)
    X = np.transpose(X, (1, 0, 2, 3))
    Y = np.array(Y, dtype=np.float32)
    Y = np.transpose(Y, (1, 0, 2))
    # 48*30(25)*10*2
    train_x = X[0:int(train_per * num_data), :, :, :]
    train_y = Y[0:int(train_per * num_data), :, :]
    test_x1 = X[int(train_per * num_data):, :, :, :]
    test_y1 = Y[int(train_per * num_data):, :, :]
    train_x1 = []
    train_y1 = []
    for i in range(0, int(train_x.shape[0] / 3)):
        for j in range(0, train_x.shape[1]):
            train_x1.append(train_x[3 * i][j])
            train_x1.append(train_x[3 * i + 1][j])
            train_x1.append(train_x[3 * i + 2][j])
            train_y1.append(train_y[3 * i][j])
            train_y1.append(train_y[3 * i + 1][j])
            train_y1.append(train_y[3 * i + 2][j])

    train_x1 = np.array(train_x1)
    train_y1 = np.array(train_y1)
    # print(train_x1.shape)
    # input()

    # # 训练集划分是否正确。是正确的
    # plt.figure()
    # for i in range(0, train_x1.shape[0]):
    #     print(i)
    #     plt.plot(train_x1[i, :, 0], train_x1[i, :, 1], 'r.', label='值x', linestyle='solid', color='green')
    #     plt.plot(train_y1[i, 0], train_y1[i, 1], 'r.', label='值y', linestyle='solid', color='red')
    #     plt.legend()
    #     plt.show()
    # input()
    # 测试集划分是否正确。是正确的
    # plt.figure()
    # for i in range(0, test_x.shape[0]):
    #     print(i)
    #     plt.plot(test_x[i, :, 0], test_x[i, :, 1], 'r.', label='值x', linestyle='solid', color='green')
    #     plt.plot(test_y[i, :, 0], test_y[i, :, 1], 'r.', label='值y', linestyle='solid', color='red')
    #     plt.legend()
    #     plt.show()
    # input()
    # # 另一个测试集也正确
    # plt.figure()
    # for i in range(0, test_x1.shape[0]):
    #     print(i)
    #     for j in range(0, test_x1.shape[1]):
    #         plt.plot(test_x1[i, j, :, 0], test_x1[i, j, :, 1], 'r.', label='值x', linestyle='solid', color='green')
    #         plt.plot(test_y1[i, j, 0], test_y1[i, j, 1], 'r.', label='值y', linestyle='solid', color='red')
    #         plt.legend()
    #         plt.show()
    # input()

    return train_x1, train_y1, test_x, test_y, X, Y   # 测试集是全集。获得更多数据。
    # return train_x1, train_y1, test_x, test_y, test_x1, test_y1  # 测试集是0.2倍全集。数据比较少。


x_train, y_train, x_test, y_test, x_test1, y_test1 = fun_data3()
feature_size = x_train.shape[-1]
embedding_dim = 32
qkv_length1 = embedding_dim
between_attention = feature_size
qkv_length2 = embedding_dim
output_dim = feature_size
# 各神经网络层定义：
WQ_t = tf.keras.layers.Dense(qkv_length1)
WK_t = tf.keras.layers.Dense(qkv_length1)
WQ_s = tf.keras.layers.Dense(qkv_length2)
WK_s = tf.keras.layers.Dense(qkv_length2)
embedding_layer = tf.keras.layers.Dense(embedding_dim)
attention_layer1 = tf.keras.layers.Attention()
dense_layer_0 = tf.keras.layers.Dense(between_attention)
attention_layer2 = tf.keras.layers.Attention()
layer_normal1 = tf.keras.layers.LayerNormalization(axis=-1)
layer_normal2 = tf.keras.layers.LayerNormalization(axis=-1)
dropout_layer1 = tf.keras.layers.Dropout(0.2)
lstm_layer = tf.keras.layers.LSTM(HIDDEN_SIZE, return_sequences=False, return_state=False)
dense_layer1 = tf.keras.layers.Dense(HIDDEN_SIZE)   # 本层输出维数没什么要求，就先设为这个数了
dense_layer2 = tf.keras.layers.Dense(output_dim)

# 神经网络连接结构：
# 下一行括号中最后一个数是特征维数。本来应该设置成3维，但它会自动加上一维，在最前面，表示batchsize，所以定义时只需设成2维。
# 3*10*2
input_tensor = tf.keras.Input(shape=(None, feature_size))
embedding = embedding_layer(input_tensor)
embedding_t = tf.transpose(embedding, perm=[1, 0, 2])

# s_attention
qs = WQ_s(embedding_t)
ks = WK_s(embedding_t)
atten2 = attention_layer2([qs, ks])
norm_data_t = layer_normal2(atten2 + embedding_t)
norm_data_1 = tf.transpose(norm_data_t, perm=[1, 0, 2])

# norm_data_1 = embedding

# t_attention
# q.shape: (batch_size, Timestep, qkv_length1)  3*10*2
qt = WQ_t(norm_data_1)
kt = WK_t(norm_data_1)
# attention_outputs.shape :(batch_size, Timestep, qkv_length1)  3*10*2
atten1 = attention_layer1([qt, kt])
# norm_data.shape: (batch_size, Timestep, embeddingdim)
norm_data_2 = layer_normal1(atten1 + norm_data_1)
hidden_tensor1 = lstm_layer(norm_data_2)

# hidden_tensor1 = lstm_layer(norm_data_1)
# hidden_tensor1 = lstm_layer(embedding)
hidden_tensor2 = dense_layer1(hidden_tensor1)
hidden_tensor3 = tf.math.tanh(hidden_tensor2)
output_tensor = dense_layer2(hidden_tensor3)

# 创建模型
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error',  # 损失函数用均方误差
              )

history = model.fit(x_train, y_train,
                    # 每次喂入神经网络的样本数
                    batch_size=BATCH_SIZE,
                    # 数据集的迭代次数
                    epochs=epochs,
                    # validation_data=(x_test, y_test),
                    # 每多少次训练集迭代，验证一次测试集
                    # validation_freq=validation_freq,
                    # callbacks=[cp_callback]
                    )



# 绘图函数
def print_history(history):
    # print(history.history)
    # input()
    # plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train_loss')
    plt.legend()
    plt.show()


# 真实
def predict(input_seq):
    # print(input_seq.shape)
    # input()
    input_seq = np.expand_dims(input_seq, axis=0)
    out = []
    while True:
        predicted_value = model.predict(input_seq)
        out.append(predicted_value)
        predicted_value = np.expand_dims(predicted_value, axis=0)
        input_seq = np.concatenate((input_seq[:, 1:, :], predicted_value), axis=1)
        if len(out) >= TIME_STEPS:   # 控制输出的长度。随便设。
            break
    out = np.array(out, dtype=np.float32)
    return out


# 理想
def predict2(input_seq):
    # 24*10*2
    # print(input_seq.shape)
    # input()
    out = []

    for i in range(input_seq.shape[0] - TIME_STEPS, input_seq.shape[0]):
        predicted_value = model.predict(np.expand_dims(input_seq[i], axis=0))
        # print(predicted_value.shape)
        out.append(predicted_value)  # 长为循环时间步数（10）。

    #     plt.plot(predicted_value[:, 0], predicted_value[:, 1], 'r.', label='预测', linestyle='solid', color='green')
    #     plt.plot(input_seq[i, :, 0],
    #              input_seq[i, :, 1],
    #              'r.',
    #              label='真实数据',
    #              linestyle='dotted',
    #              color='red')
    #     plt.legend()
    #     plt.show()
    # input()

    out = np.array(out, dtype=np.float32)  # 10*1*2
    return out


# 输出模型各层的参数状况
model.summary()

'''
# 预测（真实情况）
plt.figure()
# 反归一
y_test_shape = y_test.shape  # 12*10*2
y_test = np.reshape(y_test, (y_test_shape[0] * y_test_shape[1], y_test_shape[2]))
y_test = sc1.inverse_transform(y_test)
y_test = np.reshape(y_test, (y_test_shape[0], y_test_shape[1], y_test_shape[2]))
# 12*10*2
for i in range(0, x_test.shape[0]):
    predicted_value = predict(x_test[i])
    predicted_value = np.squeeze(predicted_value)
    # print(predicted_value.shape)
    # print(y_test[i].shape)

    # print(predicted_value[:, 0])
    # print(predicted_value[:, 1])
    # print(y_test[i, :, 0])
    # print(y_test[i, :, 1])
    # input()

    predicted_value = sc1.inverse_transform(predicted_value)
    x_test_real = x_test[i, :, :]  # 10*2
    x_test_real = sc1.inverse_transform(x_test_real)

    # 对预测的函数曲线进行绘图。
    plt.plot(x_test_real[:, 0], x_test_real[:, 1], label='历史', linestyle='solid', color='grey')
    if i % 3 == 0:
        plt.plot(predicted_value[:, 0], predicted_value[:, 1], 'r.', label='预测', linestyle='solid', color='lightcoral')
        plt.plot(y_test[i, :, 0], y_test[i, :, 1], 'r.', label='真实数据', linestyle='dotted', color='red')
    if i % 3 == 1:
        plt.plot(predicted_value[:, 0], predicted_value[:, 1], 'r.', label='预测', linestyle='solid', color='deepskyblue')
        plt.plot(y_test[i, :, 0], y_test[i, :, 1], 'r.', label='真实数据', linestyle='dotted', color='blue')
    if i % 3 == 2:
        plt.plot(predicted_value[:, 0], predicted_value[:, 1], 'r.', label='预测', linestyle='solid', color='lime')
        plt.plot(y_test[i, :, 0], y_test[i, :, 1], 'r.', label='真实数据', linestyle='dotted', color='green')
    plt.legend()
    if i % 3 == 2:
        plt.show()

'''


def get_rmse(y_test, y_predict):
    mse = np.sum((y_test - y_predict) ** 2) / len(y_test)
    rmse = np.sqrt(mse)
    return rmse



# 预测（理想情况）
# 反归一
y_test1_shape = y_test1.shape  # 12*24*2
y_test1 = np.reshape(y_test1, (y_test1_shape[0] * y_test1_shape[1], y_test1_shape[2]))
y_test1 = sc1.inverse_transform(y_test1)
y_test1 = np.reshape(y_test1, (y_test1_shape[0], y_test1_shape[1], y_test1_shape[2]))

print_history(history)
plt.figure()
for i in range(0, x_test1.shape[0]):  # xtest1:12*24*10*2
    print(i)
    predicted_value = predict2(x_test1[i])
    predicted_value = np.squeeze(predicted_value)
    x_test_real = x_test1[i, x_test1.shape[1] - TIME_STEPS, :, :]  # 10*2

    # 反归一。 x_test1:12*24*10*2。predicted_value：10*2
    predicted_value = sc1.inverse_transform(predicted_value)
    x_test_real = sc1.inverse_transform(x_test_real)

    # 均方根误差
    error = []
    for j in range(0, len(predicted_value)):
        y1 = y_test1[i, y_test1.shape[1] - TIME_STEPS:y_test1.shape[1] - (TIME_STEPS - j - 1), 0]
        y2 = y_test1[i, y_test1.shape[1] - TIME_STEPS:y_test1.shape[1] - (TIME_STEPS - j - 1), 1]
        p1 = predicted_value[0:j + 1, 0]
        p2 = predicted_value[0:j + 1, 1]
        re = (get_rmse(y1, p1) + get_rmse(y2, p2)) / 2
        error.append(re)
    plt.xlabel('预测步数')
    plt.ylabel('RMSE值')
    plt.plot(error, label='RMSE')
    plt.legend()
    plt.show()
    # input()

    # 对预测的函数曲线进行绘图。
    # y_start = -50
    # y_end = 100
    # plt.vlines(4.5, y_start, y_end, colors='lightgrey', linestyles="dashed")
    # plt.vlines(8, y_start, y_end, colors='lightgrey', linestyles="dashed")
    # plt.vlines(11.5, y_start, y_end, colors='lightgrey', linestyles="dashed")
    # plt.vlines(15, y_start, y_end, colors='lightgrey', linestyles="dashed")
    # plt.vlines(18.5, y_start, y_end, colors='lightgrey', linestyles="dashed")
    plt.xlabel('水平方向距离(m)')
    plt.ylabel('前进方向距离(m)')
    if i % 3 == 0:
        plt.plot(x_test_real[:, 0],
                 x_test_real[:, 1],
                 'r.',
                 label='历史',
                 linestyle='solid',
                 color='grey')
    else:
        plt.plot(x_test_real[:, 0],
                 x_test_real[:, 1],
                 'r.',
                 linestyle='solid',
                 color='grey')

    if i % 3 == 0:  # ytest1:12*24*2
        plt.plot(predicted_value[:, 0], predicted_value[:, 1], 'r.', label='预测数据', linestyle='solid', color='red')
        plt.plot(y_test1[i, y_test1.shape[1] - TIME_STEPS - 1:y_test1.shape[1], 0],
                 y_test1[i, y_test1.shape[1] - TIME_STEPS - 1:y_test1.shape[1], 1],
                 'r.',
                 label='真实数据',
                 linestyle='dotted',
                 color='blue')
    if i % 3 == 1:
        plt.plot(predicted_value[:, 0], predicted_value[:, 1], 'r.', linestyle='solid', color='red')
        plt.plot(y_test1[i, y_test1.shape[1] - TIME_STEPS - 1:y_test1.shape[1], 0],
                 y_test1[i, y_test1.shape[1] - TIME_STEPS - 1:y_test1.shape[1], 1],
                 'r.',
                 # label='真实数据',
                 linestyle='dotted',
                 color='blue')
    if i % 3 == 2:
        plt.plot(predicted_value[:, 0], predicted_value[:, 1], 'r.', linestyle='solid', color='red')
        plt.plot(y_test1[i, y_test1.shape[1] - TIME_STEPS - 1:y_test1.shape[1], 0],
                 y_test1[i, y_test1.shape[1] - TIME_STEPS - 1:y_test1.shape[1], 1],
                 'r.',
                 # label='真实数据',
                 linestyle='dotted',
                 color='blue')
    plt.legend()
    if i % 3 == 2:
        plt.show()
