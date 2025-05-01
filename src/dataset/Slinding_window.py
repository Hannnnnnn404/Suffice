import pandas as pd
import numpy as np
import torch
from datetime import timedelta

def process_streaming_data(updates_root, window_size):
    """
    :param vector_root:
    :param updates_root:
    :param window_size:
    :return: 每滑动一次窗口，vector对应的value的update [W0, W1, W2, ..., Wn]
    """
    # 加载初始
    initial_path = f'{updates_root}/initial_vectors.pt'
    # 1. 读取整个张量
    tensor_vector = torch.load(initial_path)

    # 2. 逐行存入 list
    vectors = [tensor_vector[i] for i in range(tensor_vector.shape[0])]

    # 3. 检查结果
    print(f"总共 {len(vectors)} 个行向量")
    print("第一个行向量:", vectors[0])  # 示例：输出第一行

    # 加载更新记录并设置日期类型
    updates_path = f'{updates_root}/updates_by_timestamp.csv'
    updates_df = pd.read_csv(updates_path)
    updates_df['date'] = pd.to_datetime(updates_df['date'])
    updates_df.set_index('date', inplace=True)

    # 保存每个窗口的更新和过期信息
    windows_updates = {}

    # 追踪评论的活动状态
    active_behavior = pd.DataFrame(columns=['id', 'date', 'row_index', 'col_index'])

    # 窗口滑动处理
    start_date = updates_df.index.min()
    end_date = updates_df.index.max()

    # 处理每天的数据，逐步构建窗口
    current_date = start_date
    window = 0
    while current_date <= end_date:
        window_end = min(current_date + pd.Timedelta(days=window_size - 1), end_date)
        window += 1
        print("window", window, ": start", current_date)
        if current_date == start_date:
            window_data = updates_df.loc[current_date:window_end]

            # 添加新行为到活动行为
            new_entries = window_data.copy()
            new_entries.loc[:, 'behave'] = 1  # 标记为新加入
            windows_updates[window] = new_entries
            # 更新活动评论记录
            active_behavior = pd.concat([active_behavior, window_data.reset_index()], ignore_index=True)

        else:
            # 过期处理：移除窗口开始前的评论
            expired_date = current_date - pd.Timedelta(days=1)  # 窗口之前过期
            expired_behavior = pd.DataFrame(columns=['id', 'date', 'row_index', 'col_index'])
            if expired_date >= start_date:
                expired_behavior = active_behavior[active_behavior['date'] == expired_date].copy()
                # 标记为过期
                if not expired_behavior.empty:
                    expired_behavior.loc[:, 'behave'] = -1

            # 新增处理：增加窗口新增的评论, 新增timestamp就是步长
            window_data = pd.DataFrame()
            if window_end in updates_df.index:
                window_data = updates_df.loc[[window_end]]
                insert_behavior = window_data.copy().reset_index()
                # 标记为新加入
                insert_behavior.loc[:, 'behave'] = 1

                new_entries = pd.concat([expired_behavior, insert_behavior], ignore_index=True)
                windows_updates[window] = new_entries

            # 更新活动评论记录
            active_behavior = pd.concat([active_behavior, window_data])

        # 移动到下一天
        current_date += pd.Timedelta(days=1)

    return windows_updates, vectors


def process_streaming_syn_data(vector_root, updates_root, window_size):
    """
    :param vector_root:
    :param updates_root:
    :param window_size:
    :return: 每滑动一次窗口，vector对应的value的update [W0, W1, W2, ..., Wn]
    """
    # 加载初始
    initial_path = f'{vector_root}'
    # 1. 读取整个张量
    tensor_vector = torch.load(initial_path)

    # 2. 逐行存入 list
    vectors = [tensor_vector[i] for i in range(tensor_vector.shape[0])]

    # 3. 检查结果
    print(f"总共 {len(vectors)} 个行向量")
    print("第一个行向量:", vectors[0])  # 示例：输出第一行

    # 加载更新记录并设置日期类型
    updates_path = f'{updates_root}'
    updates_df = pd.read_csv(updates_path)
    updates_df['date'] = pd.to_datetime(updates_df['date'])  # 确保日期是 datetime 类型
    updates_df.set_index('date', inplace=True)

    # 保存每个窗口的更新和过期信息
    windows_updates = {}

    # 追踪评论的活动状态
    active_behavior = pd.DataFrame(columns=['id', 'date', 'row_index', 'col_index'])

    # 窗口滑动处理
    start_date = updates_df.index.min()
    end_date = updates_df.index.max()

    # 处理每天的数据，逐步构建窗口
    current_date = start_date
    window = 0
    while current_date <= end_date:
        window_end = min(current_date + pd.Timedelta(days=window_size - 1), end_date)
        window += 1
        print("window", window, ": start", current_date)
        if current_date == start_date:
            window_data = updates_df.loc[current_date:window_end]

            # 添加新行为到活动行为
            new_entries = window_data.copy()
            new_entries.loc[:, 'behave'] = 1  # 标记为新加入
            windows_updates[window] = new_entries
            # 更新活动评论记录
            active_behavior = pd.concat([active_behavior, window_data.reset_index()], ignore_index=True)

        else:
            # 过期处理：移除窗口开始前的评论
            expired_date = current_date - pd.Timedelta(days=1)  # 窗口之前过期
            expired_behavior = pd.DataFrame(columns=['id', 'date', 'row_index', 'col_index'])
            if expired_date >= start_date:
                expired_behavior = active_behavior[active_behavior['date'] == expired_date].copy()
                # 标记为过期
                if not expired_behavior.empty:
                    expired_behavior.loc[:, 'behave'] = 0

            # 新增处理：增加窗口新增的评论, 新增timestamp就是步长
            window_data = pd.DataFrame()
            if window_end in updates_df.index:
                window_data = updates_df.loc[[window_end]]
                insert_behavior = window_data.copy().reset_index()
                # 标记为新加入
                insert_behavior.loc[:, 'behave'] = 1

                new_entries = pd.concat([expired_behavior, insert_behavior], ignore_index=True)
                windows_updates[window] = new_entries

            # 更新活动评论记录
            active_behavior = pd.concat([active_behavior, window_data])

        # 移动到下一天
        current_date += pd.Timedelta(days=1)

    return windows_updates, vectors


def process_streaming_data_over(updates_root, window_size, total_window_target):

    initial_path = f'{updates_root}/initial_vectors.pt'
    tensor_vector = torch.load(initial_path)
    vectors = [tensor_vector[i] for i in range(tensor_vector.shape[0])]

    print(f"总共 {len(vectors)} 个行向量")
    print("第一个行向量:", vectors[0])

    # 加载更新记录
    updates_path = f'{updates_root}/updates_by_timestamp.csv'
    updates_df = pd.read_csv(updates_path)
    updates_df['date'] = pd.to_datetime(updates_df['date'])
    updates_df.set_index('date', inplace=True)

    unique_dates = sorted(updates_df.index.unique())
    date_to_data = {date: updates_df.loc[[date]] for date in unique_dates}

    windows_updates = {}
    start_date = updates_df.index.min()
    original_days = len(unique_dates)

    current_date = start_date
    window = 0

    while window < total_window_target:
        window += 1
        print(f"window {window}: start {current_date}")

        window_range = pd.date_range(start=current_date, periods=window_size)

        window_data_list = []
        for i, d in enumerate(window_range):
            offset = (d - start_date).days
            mapped_date = unique_dates[offset % original_days]
            if mapped_date in date_to_data:
                daily_data = date_to_data[mapped_date].copy()
                daily_data.index = [d] * len(daily_data)
                window_data_list.append(daily_data)

        if window_data_list:
            window_data = pd.concat(window_data_list, ignore_index=False)
            window_data['behave'] = 1
            windows_updates[window] = window_data

        current_date += timedelta(days=1)

    return windows_updates, vectors


if __name__ == '__main__':
    # 设置数据路径和窗口大小
    updates_path1 = 'Reddit'
    window_size1 = 3
    window_size2 = 10
    window_size3 = 30

    # 调用函数
    windows_update1_1 = process_streaming_data(updates_path1, window_size1)
