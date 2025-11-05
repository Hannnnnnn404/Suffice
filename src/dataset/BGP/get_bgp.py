import pandas as pd
import torch

if __name__ == '__main__':
    # read cvs download from script
    df = pd.read_csv('bgp_events_monitor0_all_events.csv', nrows=800000)

    num_unique_as_nodes = df['AS_node_id'].nunique()
    print("一 AS node ID number:", num_unique_as_nodes)

    num_unique_paths = df['path_id'].nunique()
    print("number of unique path:", num_unique_paths)

    as_path_counts = df.groupby('AS_node_id')['path_id'].nunique().reset_index()
    as_path_counts.columns = ['AS_node_id', 'num_paths']
    as_path_counts = as_path_counts.sort_values(by='num_paths', ascending=False)

    print("\nEach As Node has AS Path:")
    print(as_path_counts.head(10))

    path_as_counts = df.groupby('path_id')['AS_node_id'].nunique().reset_index()
    path_as_counts.columns = ['path_id', 'num_AS_nodes']
    path_as_counts = path_as_counts.sort_values(by='num_AS_nodes', ascending=False)
    print(path_as_counts.head(10))

    df.rename(columns={'AS_node_id': 'row', 'path_id': 'col'}, inplace=True)

    df.sort_values(by='date', inplace=True)

    df['row_index'] = pd.factorize(df['row'])[0]
    df['col_index'] = pd.factorize(df['col'])[0]

    df_vector = df[['id', 'date', 'row_index', 'col_index']]

    df_vector['date'] = pd.to_datetime(df_vector['date'], format='%Y%m%d')
    start_date = df_vector['date'].min()
    cutoff_date = start_date + pd.Timedelta(days=150)

    df_initial = df_vector[df_vector['date'] < cutoff_date]

    num_rows = df_vector['row_index'].max() + 1
    num_cols = df_vector['col_index'].max() + 1

    tensor_vector = torch.zeros((num_rows, num_cols), dtype=torch.float32)

    print("Tensor shape:", tensor_vector.shape)
    torch.save(tensor_vector, "initial_vectors.pt")

    # save DataFrame to CSV 文件
    df_vector.to_csv('updates_by_timestamp.csv', index=False)
