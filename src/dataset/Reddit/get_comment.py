import torch
from datasets import load_dataset
import pandas as pd
import numpy as np

if __name__ == '__main__':
    dataset = load_dataset("alvanlii/reddit-comments-uwaterloo", "year_2015")

    df = pd.DataFrame(dataset['train'])
    print(df.columns)

    filtered_df = df[df['poster'] != '[deleted]'][['id', 'poster', 'link_id', 'date_utc']]

    # 1. How many users
    num_users = filtered_df['poster'].nunique()

    # 2. How many comment
    num_comments = len(filtered_df)

    # 3. How many comments for each users
    comments_per_user = filtered_df['poster'].value_counts()

    filtered_df['date_utc'] = pd.to_datetime(filtered_df['date_utc'], unit='s').dt.date
    start_date = filtered_df['date_utc'].min()
    end_date = filtered_df['date_utc'].max()
    duration_days = (end_date - start_date).days

    print(f"Number of unique users: {num_users}")
    print(f"Number of comments: {num_comments}")
    print("Comments per user: ")
    print(comments_per_user)
    print(f"Time span from {start_date} to {end_date}, total {duration_days} days.")

    comment_counts = filtered_df.groupby(['link_id', 'poster']).size().reset_index(name='counts')
    multiple_comments = comment_counts[comment_counts['counts'] > 1]
    sorted_comments = comment_counts.sort_values(by='counts', ascending=False)

    print("User with the most comments on a single post:")
    print(sorted_comments.head(1))

    filtered_df.sort_values(by='date_utc', inplace=True)

    filtered_df.rename(columns={'poster': 'row', 'link_id': 'col', 'date_utc': 'date'}, inplace=True)

    df_whole = filtered_df[['id', 'date', 'row', 'col']]

    df_whole['row_index'] = pd.factorize(df_whole['row'])[0]
    df_whole['col_index'] = pd.factorize(df_whole['col'])[0]

    df_whole = df_whole[['id', 'date', 'row_index', 'col_index']]

    num_rows = df_whole['row_index'].max() + 1
    num_cols = df_whole['col_index'].max() + 1

    tensor_vector = torch.zeros((num_rows, num_cols), dtype=torch.float32)

    print("Tensor shape:", tensor_vector.shape)
    print(tensor_vector)
    torch.save(tensor_vector, "initial_vectors.pt")

    # save to csv file
    df_whole.to_csv('updates_by_timestamp.csv', index=False)
