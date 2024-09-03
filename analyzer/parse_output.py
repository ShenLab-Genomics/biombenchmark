import os
import sys
import argparse
import numpy as np
import pandas as pd


class DataProcessor:
    def __init__(self, filename):
        self.filename = filename
        self.base_name = os.path.splitext(filename)[0]
        self.output_filename = f"{self.base_name}_collected_data.csv"
        self.parameters = ['method', 'Epoch']
        self.data_frame = None

    def read_and_process_data(self):
        data_list = []

        with open(self.filename, 'r') as infile:
            for line in infile:
                if line.startswith('Epoch'):
                    self.current_epoch = line.strip()
                elif line.startswith('Test') and self.current_epoch is not None:
                    self.process_line(line, self.current_epoch, data_list)
        # 将数据列表转换为Pandas DataFrame
        self.data_frame = pd.DataFrame(data_list, columns=self.parameters)

    def process_line(self, line, epoch, data_list):
        rest_of_line = line[4:].lstrip('\t')
        parts = rest_of_line.split()
        method = parts[0]
        data_part = rest_of_line.replace(method, '', 1).strip()

        row_data = {'method': method}
        key_value_pairs = [item.split(': ')
                           for item in data_part.split('\t') if item]
        for key, value in key_value_pairs:
            key = key.strip()
            if key:
                row_data[key] = value
                if key not in self.parameters:
                    self.parameters.append(key)
        # 将当前的 Epoch 信息添加到行数据中
        row_data['Epoch'] = epoch.split(': ')[1]
        data_list.append(row_data)

    def save_data_to_csv(self, output=None):
        # 保存 DataFrame 到 CSV 文件
        if output is not None:
            self.output_filename = output
        self.data_frame.to_csv(self.output_filename, index=False)
        print(f"Data has been saved to {self.output_filename}")


def search_file(filename, search_dirs):
    # """在提供的目录列表中搜索文件"""
    for directory in search_dirs:
        potential_path = os.path.join(directory, filename)
        if os.path.isfile(potential_path):
            return potential_path
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    full_path = args.input
    if os.path.exists(full_path):
        processor = DataProcessor(full_path)
        processor.read_and_process_data()
        processor.save_data_to_csv(args.output)
        print("Data collection and processing is complete.")
    else:
        print("File not found. Please check the filename and try again.")
        sys.exit(1)
