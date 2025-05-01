#!/bin/bash

# 指定目标日期文件夹所在的根目录
root_dir="/home/hanhan/BGP_Datasets"

# 遍历根目录下的所有日期文件夹
for date_dir in "$root_dir"/*/; do
    date=$(basename "$date_dir")  # 获取日期文件夹的名称（日期部分）

    # 同时处理每个监测点 rrc00 至 rrc26
    for monitor_id in $(seq 0 26); do
        (
            # 创建输出目录（如果不存在）
            rib_output_dir="$HOME/BGP_Data_txt/$date/rib/$monitor_id"
            updates_output_dir="$HOME/BGP_Data_txt/$date/updates/$monitor_id"

            mkdir -p "$rib_output_dir" "$updates_output_dir"

            # 遍历rib目录的.gz文件并在后台执行bgpdump命令
            for file in "$date_dir"/rib/"$monitor_id"/*.gz; do
                filename=$(basename "$file")
                output_file="$rib_output_dir/${filename%.gz}.txt"
                bgpdump -M -t dump -O "$output_file" "$file" &
            done

            # 遍历updates目录的.gz文件并在后台执行bgpdump命令
            for file in "$date_dir"/updates/"$monitor_id"/*.gz; do
                filename=$(basename "$file")
                output_file="$updates_output_dir/${filename%.gz}.txt"
                bgpdump -M -t dump -O "$output_file" "$file" &
            done

            # 等待当前监测点的所有下载和转换完成
            wait
        ) &
    done
    # 等待当天所有监测点的所有处理完成
    wait
done

echo "All processing complete."
