import pandas as pd
import torch

if __name__ == '__main__':
    # read from RetailRocket events.csv
    df = pd.read_csv('events.csv')

    df = df[df['event'] == 'transaction'].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['date'] = df['timestamp'].dt.date  # 只保留日期部分（也可以用 .dt.floor('D')）

    df.rename(columns={
        'visitorid': 'user_id',
        'itemid': 'item_id'
    }, inplace=True)

    df.reset_index(drop=True, inplace=True)
    df['id'] = df.index

    df_final = df[['id', 'date', 'user_id', 'item_id']]

    df_final.sort_values(by='date', inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    df_final.to_csv('retailrocket_user_item_purchases.csv', index=False)

    print(f"提取完成，共 {len(df_final)} 条购买事件。")
    print(df_final.head())

    # 1. number of users
    num_users = df_final['user_id'].nunique()

    # 2. number of items
    num_items = df_final['item_id'].nunique()

    user_item_counts = df_final.groupby('user_id')['item_id'].nunique().reset_index()
    user_item_counts.columns = ['user_id', 'num_unique_items']

    item_user_counts = df_final.groupby('item_id')['user_id'].nunique().reset_index()
    item_user_counts.columns = ['item_id', 'num_unique_users']

    start_date = df_final['date'].min()
    end_date = df_final['date'].max()

    print(f"{num_users}")
    print(f"{num_items}")
    print(f"{start_date} ~ {end_date}")

    user_item_counts_sorted = user_item_counts.sort_values(by='num_unique_items', ascending=False)
    print(user_item_counts_sorted.head(10))

    item_user_counts_sorted = item_user_counts.sort_values(by='num_unique_users', ascending=False)
    print(item_user_counts_sorted.head(10))

    df = pd.read_csv('retailrocket_user_item_purchases.csv')

    df['date'] = pd.to_datetime(df['date'])

    # rename column
    df.rename(columns={'user_id': 'row', 'item_id': 'col'}, inplace=True)

    df_whole = df[['id', 'date', 'row', 'col']].copy()
    df_whole['row_index'] = pd.factorize(df_whole['row'])[0]
    df_whole['col_index'] = pd.factorize(df_whole['col'])[0]
    df_whole = df_whole[['id', 'date', 'row_index', 'col_index']]

    start_date = df_whole['date'].min()
    cutoff_date = start_date + pd.Timedelta(days=150)

    df_initial = df_whole[df_whole['date'] < cutoff_date]
    df_update = df_whole[df_whole['date'] >= cutoff_date]

    num_rows = df_whole['row_index'].max() + 1
    num_cols = df_whole['col_index'].max() + 1

    tensor_vector = torch.zeros((num_rows, num_cols), dtype=torch.float32)

    # row_indices = torch.tensor(df_initial['row_index'].values, dtype=torch.long)
    # col_indices = torch.tensor(df_initial['col_index'].values, dtype=torch.long)
    #
    # tensor_vector.index_put_((row_indices, col_indices), torch.ones_like(row_indices, dtype=torch.float32),
    #                          accumulate=True)

    # ====== save to csv ======
    print("初始 Tensor 维度:", tensor_vector.shape)
    torch.save(tensor_vector, "initial_vectors.pt")

    df_whole.to_csv("updates_by_timestamp.csv", index=False)
