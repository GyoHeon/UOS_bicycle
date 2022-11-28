'''
2~6월달의 로우 데이터를
시간에 따라 분류하거나 자전거 번호등의 사용하지 않는 데이터를 제거하였음.
'''

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

'''
Data 정리해서 npy로 저장
'''

# station data load
station = pd.read_csv('../data/station(21.01.31).csv')

# 계산용 numpy array
station_array = np.array(station.iloc[:, 0])

# use data load
data_2 = pd.read_csv('../raw_data/2021.02.csv', sep = ',', encoding = 'cp949')
data_3 = pd.read_csv('../raw_data/2021.03.csv', sep = ',', encoding = 'cp949')
data_4 = pd.read_csv('../raw_data/2021.04.csv', sep = ',', encoding = 'cp949')
data_5 = pd.read_csv('../raw_data/2021.05.csv', sep = ',', encoding = 'cp949')
data_6 = pd.read_csv('../raw_data/2021.06.csv', sep = ',', encoding = 'cp949')

'''
밑에서 부터는 np.array 데이터를 사용하였음.
'''

# 사용하지 않는 column 제거
data_2 = data_2.drop(columns=['자전거번호', '대여 대여소명', '대여거치대',
                    '반납대여소명', '반납거치대'], axis=1)
data_3 = data_3.drop(columns=['자전거번호', '대여 대여소명', '대여거치대',
                    '반납대여소명', '반납거치대'], axis=1)
data_4 = data_4.drop(columns=['자전거번호', '대여 대여소명', '대여거치대',
                    '반납대여소명', '반납거치대'], axis=1)
data_5 = data_5.drop(columns=['자전거번호', '대여 대여소명', '대여거치대',
                    '반납대여소명', '반납거치대'], axis=1)
data_6 = data_6.drop(columns=['자전거번호', '대여 대여소명', '대여거치대',
                    '반납대여소명', '반납거치대'], axis=1)

# datafram to numpy array
data_2 = data_2.to_numpy()
data_3 = data_3.to_numpy()
data_4 = data_4.to_numpy()
data_5 = data_5.to_numpy()
data_6 = data_6.to_numpy()

'''
가지고 있는 node 외에 다른 node를 표시하는 데이터는 제거하였음.
제거되는 데이터의 수가 많지만 추가로 node의 정보를 확인할 수는 없었음.
'''

# 출발, 도착 정류소의 위치 좌표가 있는지 True, False 값으로 반환
start_2 = np.isin(data_2[:, 1], station_array)
end_2 = np.isin(data_2[:, 3], station_array)
start_3 = np.isin(data_3[:, 1], station_array)
end_3 = np.isin(data_3[:, 3], station_array)
start_4 = np.isin(data_4[:, 1], station_array)
end_4 = np.isin(data_4[:, 3], station_array)
start_5 = np.isin(data_5[:, 1], station_array)
end_5 = np.isin(data_5[:, 3], station_array)
start_6 = np.isin(data_6[:, 1], station_array)
end_6 = np.isin(data_6[:, 3], station_array)

# 출발, 도착 정류소 모두 있는 경우 1 반환
real_2 = start_2 * end_2
real_3 = start_3 * end_3
real_4 = start_4 * end_4
real_5 = start_5 * end_5
real_6 = start_6 * end_6

# 필요없는 데이터 index
index_f_2 = np.where(real_2 == 0)[0]
index_f_3 = np.where(real_3 == 0)[0]
index_f_4 = np.where(real_4 == 0)[0]
index_f_5 = np.where(real_5 == 0)[0]
index_f_6 = np.where(real_6 == 0)[0]

# 필요없는 데이터 삭제한 데이터
real_station_2 = np.delete(data_2, index_f_2, 0)
real_station_3 = np.delete(data_3, index_f_3, 0)
real_station_4 = np.delete(data_4, index_f_4, 0)
real_station_5 = np.delete(data_5, index_f_5, 0)
real_station_6 = np.delete(data_6, index_f_6, 0)

'''
날짜, 시간으로 데이터 나누기
'''

# 따릉이 데이터
data_2 = np.load('../raw_data/data_2.npy', allow_pickle = True)
data_3 = np.load('../raw_data/data_3.npy', allow_pickle = True)
data_4 = np.load('../raw_data/data_4.npy', allow_pickle = True)
data_5 = np.load('../raw_data/data_5.npy', allow_pickle = True)
data_6 = np.load('../raw_data/data_6.npy', allow_pickle = True)

# 주말 체크 (0~4: 평일(0), 5~6: 주말(1))
def is_weekend(x):
    if x>4:
        return 0
    else:
        return 1

# 주말, 평일 데이터 나누기
weekday_list = []
weekend_list = []

for i in data_6:
    if is_weekend(datetime.strptime(i[0][:10], '%Y-%m-%d').weekday()) == True:
        weekday_list.append(i)
    else:
        weekend_list.append(i)

weekday_list = np.array(weekday_list)
weekend_list = np.array(weekend_list)

# 평일 따릉이
data_2 = np.load('../raw_data/data_2_weekday.npy', allow_pickle = True)
data_3 = np.load('../raw_data/data_3_weekday.npy', allow_pickle = True)
data_4 = np.load('../raw_data/data_4_weekday.npy', allow_pickle = True)
data_5 = np.load('../raw_data/data_5_weekday.npy', allow_pickle = True)
data_6 = np.load('../raw_data/data_6_weekday.npy', allow_pickle = True)

# 출퇴근 시간 함수 (시간은 임의로 3시간 정함)
def gotowork(x):
    if x == 7 or x == 8 or x == 9:
        return 1
    elif x == 17 or x == 18 or x == 19:
        return 2

'''
출근, 퇴근 데이터 나누기
출근:7~10시
퇴근:17~20시
'''

on_list = []
off_list = []

for i in data_6:
    if gotowork(datetime.strptime(i[0][11:], '%H:%M:%S').hour) == 1:
        on_list.append(i)
    elif gotowork(datetime.strptime(i[0][11:], '%H:%M:%S').hour) == 2:
        off_list.append(i)

on_list = np.array(on_list)
off_list = np.array(off_list)