import os
import sys
import argparse
import numpy as np
import pandas as pd


import argparse
import sys
import numpy as np
import os
import pandas as pd
import re


class DataProcessor:
    # 分为以下几块：初始化，预处理（筛选行，读取组别），数据处理，数据写入，数据保存
    def __init__(self, filename):
        self.filename = filename
        self.base_name = os.path.splitext(filename)[0]
        self.output_filename = f"{self.base_name}_collected_data.csv"
        self.data_frame = None
        self.df = pd.DataFrame()

    def preprogress(self):
        with open(self.filename, 'r') as data:
            lines = data.readlines()
        # 读取行，找到以Epoch（小组别）开头的行
        Methods = []
        data_list = []
        parameters = ['Method', 'Epoch']
        n = 4

        # processline(lines[]# from "Epoch" to end, Methods , data_list, parameters)
        # featureline = lines[n+1]
        for line_index, line in enumerate(lines):
            if line.startswith('Epoch:'):
                try:
                    self.processline(
                        lines[line_index: line_index + n],  data_list, parameters)
                except IndexError:
                    # 如果出现索引超出范围的错误，打印错误信息并继续处理下一条记录
                    print(
                        f"Skipping data due to index out of range at line {line_index}")
                    continue  # 继续下一次循环"""
        # 将数据列表转换为Pandas DataFrame
        self.data_frame = pd.DataFrame(data_list, columns=parameters)

    def processline(self, line_list, data_list, parameters):

        subfeature = line_list[1]

        # 处理准确率等数值
        for featureline in line_list[2:]:
            # 标注Epoch序号
            Epoch = line_list[0].split(': ')[-1].strip("\n")
            row_data = {'Epoch': Epoch}
            try:
                # 去除分号和换行符
                featureline = re.sub(r'[;\n]', ' ', str(featureline))
                # 每行第二项是Method，第三项是Stage，第四项开始是各个指标
                features = featureline.split("\t")
                row_data['Method'] = features[1]
            except IndexError:
                print(line_list)
            row_data['Stage'] = features[2]
            evaluate_set = features[3:]
            for feature in evaluate_set:
                key, value = feature.split(':')
                row_data[key] = value.strip()
            data_list.append(row_data)
            # 更新参数列表
            parameters.extend(key for key in row_data if key not in parameters)

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
        processor.preprogress()
        processor.save_data_to_csv(args.output)
        print("Data collection and processing is complete.")
    else:
        print("File not found. Please check the filename and try again.")
        sys.exit(1)
