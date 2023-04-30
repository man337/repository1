'''
处理NGSIM数据集的代码
'''
import pandas as pd
import numpy as np
import os
import random
import json

#df = pd.read_csv('data/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv',
init_df = pd.read_csv(
                'data/only_id2.csv',
                usecols=[
                           "Vehicle_ID",
                           "Frame_ID",
                           #"Total_Frames",
                           "Global_Time",
                           "Global_X",
                           "Global_Y",
                           "Local_X",
                           "Local_Y",
                           "v_Class",
                           "Location",
                           "v_length",
                           "v_Width",
                           "v_Vel",
                           "Lane_ID"
                         ],
                dtype={'Local_X': np.float64,
                       'Local_Y': np.float64,
                       'Global_X': np.float64,
                       'Global_Y': np.float64,
                       "Global_Time": np.int64
                       },
)
print('处理过的数据集读取完成')
# 若换成原数据集，则将三引号注释的代码取消注释（共2段）
# print('原数据集读取完成')
# print(init_df)
# print(init_df.info())


# ignore_index=True，则本函数会修改索引。可按需修改
def sort_by_time(df):
    return df.sort_values(by='Global_Time', ascending=True, ignore_index=True)


# 这个数据集不同车辆可以对应同一ID，还得自己改
def set_correct_ID(df=None, minid=0, maxid=0):
    next_new_id = maxid + 1
    for i in range(minid, maxid+1):
        Vehicle_tensor = df[df['Vehicle_ID'] == i]
        repeat = 0
        if Vehicle_tensor.shape[0] <= 1:
            continue
        else:
            Vehicle_tensor = Vehicle_tensor.reset_index(drop=False)
            this_v1 = Vehicle_tensor['v_length'][0]
            this_v2 = Vehicle_tensor['v_Width'][0]
            for j in range(1, Vehicle_tensor.shape[0]):
                next_v1 = Vehicle_tensor['v_length'][j]
                next_v2 = Vehicle_tensor['v_Width'][j]
                if (next_v1 != this_v1) or (next_v2 != this_v2):
                    df.loc[Vehicle_tensor['index'][j], 'Vehicle_ID'] = next_new_id
                    next_new_id += 1
                    repeat += 1
                this_v1 = next_v1
                this_v2 = next_v2
        # print("{}:重复{}".format(i, repeat))
        # if i % 100 == 0:
        #     print("{}/{}}".format(i, maxid))
    return df, next_new_id-1


def cutbyRoad(df=None, road=None):
    '''
    :param df: 打开后文件
    :param road: 路段名称
    :return: 切路df,按照全局时间排序
    '''
    road_df = df[df['Location'] == road]
    return road_df.sort_values(by='Global_Time', ascending=True, ignore_index=False)


def unitConversion(df):
    '''
    转换后长度单位为m，时间单位为0.1秒
    :param df: 被转换df
    :return: 转换后df
    '''
    ft_to_m = 0.3048
    df['Global_Time'] = df['Global_Time'] / 100
    for strs in ["Global_X", "Global_Y", "Local_X", "Local_Y", "v_length", "v_Width"]:
        df[strs] = df[strs] * ft_to_m
    # df["v_Vel"] = df["v_Vel"] * ft_to_m*3.6
    df["v_Vel"] = df["v_Vel"] * ft_to_m
    return df


def cutbyPosition(road_df, start_y=0, start_time=0, area_length=0):
    '''
    给定起始时间，起始y，区间长度，输出区间内车辆list
    :param road_df:限定路段后的df
    :param start_y: 区域开始段，单位为m
    :param start_time: 起始时间，0.1s
    :param area_length: 区域长度单位为m
    :return: vehicle_list为起始框内部车辆编号
    467m,2772s
    '''
    # 1秒之内路过的起点area_length米的车辆被记录
    area_df = road_df[(road_df['Global_Time'] >= start_time) & (road_df['Global_Time'] <= start_time+10)]
    area_df = area_df[(area_df['Global_Y'] - start_y <= area_length) & (area_df['Global_Y']-start_y >= 0)]
    vehicle_list = area_df['Vehicle_ID'].unique()  # 重复不算
    final_list = []
    lane_list = []
    for v in vehicle_list:
        a_car = area_df[(area_df['Vehicle_ID'] == v)]
        a_car = a_car.reset_index(drop=True)
        # print(a_car)
        # input()
        ini_lane = a_car.loc[0, 'Lane_ID']
        if ini_lane in lane_list:
            continue
        else:
            final_list.append(v)
            lane_list.append(ini_lane)

    if len(list(final_list)) <= 2:
        return None
    else:
        return list(final_list)


def cutbyTime(road_df, start_time=0, vehicle_list=None, time_length=0.0, stride=0.0):
    '''
    :param inint_df:road_df
    :param start_frame: 开始帧
    :param time_length: 采样时间长度,单位为s
    :param stride: 采样步长
    :return: 返回一组清洗完数据time
    '''
    temp_df = road_df[road_df['Vehicle_ID'].isin(vehicle_list)]
    one_sequence = pd.DataFrame()
    vehicle_count = 0
    for vehicle in vehicle_list:
        if_long = 0  # 在20（10）秒的最后3秒内是否有记录。
        wt = 0  # 不动的时间
        # 不动的时间是否太长
        for time in range(int(time_length * 10 / stride)):
            df = temp_df[
                (temp_df['Vehicle_ID'] == vehicle) & (temp_df['Global_Time'] == (start_time + time * stride))]
            df1 = df.reset_index(drop=False)
            if df.shape[0] == 1:
                if df1['v_Vel'][0] == 0.0:
                    wt += 1
                if wt >= 3:
                    break
        if wt >= 3:
            continue
        # 最后3秒是否还存在
        for time in range(int(time_length * 10 / stride) - 30, int(time_length * 10 / stride)):
            df = temp_df[
                (temp_df['Vehicle_ID'] == vehicle) & (temp_df['Global_Time'] == (start_time + time * stride))]
            if df.shape[0] == 1:
                if_long = 1
                break

        if if_long == 1:
            vehicle_count += 1
            for time in range(int(time_length * 10 / stride)):
                df = temp_df[
                    (temp_df['Vehicle_ID'] == vehicle) & (temp_df['Global_Time'] == (start_time + time * stride))]
                if df.shape[0] == 1:
                    one_sequence = pd.concat([one_sequence, df])
    if vehicle_count <= 2:
        return None
    return one_sequence


def saveCsv(df, file_name):
    '''
    :param df: 存入df
    :param file_name: 文件名
    :return: 无
    '''
    df = df.reset_index(drop=True)  #重置索引
    if not os.path.exists('\\'):
        os.makedirs('\\')
    df.to_csv('data\\' + file_name + '.csv', mode='a', header=True)


with open("data/dataExecute.json", "r") as f:
    conf = json.load(f)
road_df = init_df
'''
road_df = cutbyRoad(init_df, road=conf['road'])
road_df = unitConversion(road_df)
'''

min_Global_Y, max_Global_Y = road_df['Global_Y'].min(), road_df['Global_Y'].max()   # 之前是约461米。这个基本是斜着的，应该乘以根号2才是竖直距离。
min_Global_Time, max_Global_Time = road_df['Global_Time'].min(), road_df['Global_Time'].max()  # 共2772.1秒
min_Vehicle_ID, max_Vehicle_ID = road_df['Vehicle_ID'].min(), road_df['Vehicle_ID'].max()  # 1-3109
'''
road_df, max_Vehicle_ID = set_correct_ID(road_df, min_Vehicle_ID, max_Vehicle_ID)
saveCsv(road_df, file_name='only_id2')
print('ID修改完成')
'''
# print(max_Vehicle_ID)
# print(type(min_Global_Time))   # 咋变float64了

total_dist = int((max_Global_Y - min_Global_Y) / 20)  # 真实间隔20*1.414米。约23组
total_time = int((max_Global_Time - min_Global_Time) / (conf['time_step'] * 10))   # 间隔30秒。约93组
print("距离采样组数为：{}，时间采样组数为：{}。距离采样间隔为{}m，时间采样间隔为{}s".format(total_dist, total_time, conf['area_step'], conf['time_step']))

total_data = 0
# rub = 0
# 循环采集数据。外循环距离递增
for dist_index in range(conf["hist_dist"], total_dist):        # history distance
    print('距离序号{}/{}'.format(dist_index, total_dist))
    miss1 = 0
    miss2 = 0
    # 内循环时间递增
    for time_index in range(conf["hist_time"], total_time):
        if conf["noise"]:
            time_noise = random.randint(0, 30)
            #dist_noise=random.randint(0,100)
            dist_noise = 0
        else:
            time_noise, dist_noise = 0, 0

        start_time = min_Global_Time + time_index * conf['time_step'] * 10+time_noise*10
        start_y = min_Global_Y + dist_index * conf['area_step']+dist_noise
        # 初始检测距离长60米
        vehicle_list = cutbyPosition(road_df, start_y=start_y, start_time=start_time,
                                     area_length=50)

        if vehicle_list is None:
            # print('{}秒时刻，{}为起点区域50m内车辆过少，进入下个时段'.format(start_time * 0.1, start_y))
            # print('起始位置编号{}，时间编号{}，50m内车辆过少，进入下个时段'.format(dist_index, time_index))
            miss1 += 1
            continue
        # 时长20。需要调整。
        one_sequence = cutbyTime(road_df, start_time=start_time, vehicle_list=vehicle_list,
                                 time_length=20,
                                 stride=conf['stride'])
        if one_sequence is None:
            # print('{}时刻，{}为起点区域内车辆存在消失，进入下个时段'.format(start_time * 0.1, start_y))
            print('起始位置编号{}，时间编号{}，符合条件车辆不够，进入下个时段'.format(dist_index, time_index))
            miss2 += 1
        else:
            total_data += 1
            saveCsv(one_sequence, file_name='mainpig')
            print('起始位置编号{}/{}，时间编号{}/{} saved! Exist {} data! '.format(dist_index, total_dist, time_index, total_time, total_data))
        if total_data == conf["need_num"]:
            print("数据采集完成")
            break
    print('1s内，60m距离内车辆少于3辆：{}  ,20s的最后3秒内剩余车辆不足3辆或停车太多{}'.format(miss1, miss2))
    if total_data ==conf["need_num"]:
        # print("数据采集完成")
        break
