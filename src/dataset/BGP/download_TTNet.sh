#!/bin/bash

# 设置起始日期和结束日期
start_date="2004-12-14"
end_date="2005-02-14"

# 设置基本URL模板
base_url="https://data.ris.ripe.net"

# 根文件夹路径
root_dir="/home/hanhan/BGP_Datasets"

# 使用日期循环处理每一天
current_date=$(date -d "$start_date" +%Y-%m-%d)
while [ "$current_date" != $(date -d "$end_date + 1 day" +%Y-%m-%d) ]; do
    date_pattern=$(date -d "$current_date" +%Y%m%d)
    month_folder=$(date -d "$current_date" +%Y.%m)

    # 同时处理所有监测点
    for monitor_id in $(seq 0 26); do
        monitor=$(printf "rrc%02d" $monitor_id)
        
        # 同时启动RIB和Updates文件的下载
        mkdir -p "$root_dir/$date_pattern/rib/$monitor_id" &
        mkdir -p "$root_dir/$date_pattern/updates/$monitor_id" &
        for time in 0000 0800 1600; do
            wget -P "$root_dir/$date_pattern/rib/$monitor_id" "$base_url/$monitor/$month_folder/bview.$date_pattern.$time.gz" &
        done
        current_time=0
        while [ $current_time -lt 2400 ]; do
            time_string=$(printf "%04d" $current_time)
            wget -P "$root_dir/$date_pattern/updates/$monitor_id" "$base_url/$monitor/$month_folder/updates.$date_pattern.$time_string.gz" &
            # 更新时间
            if [ "${time_string:2:2}" == "55" ]; then
                current_time=$(($current_time + 45))
            else
                current_time=$(($current_time + 5))
            fi
        done
    done
    wait # 等待当天所有监测点的所有下载完成
    current_date=$(date -d "$current_date + 1 day" +%Y-%m-%d)
done
wait # 确保所有后台进程都已完成
