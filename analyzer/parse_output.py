import argparse
import sys
import numpy as np
import os
import pandas as pd
import re
import ast
import json
from torch import tensor


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

        # 若行为：Epoch: ：向上找到不包含:,\的行，读取为方法Method，对所有Epoch行的下两行进行处理，读取参数和值
        for line_index, line in enumerate(lines):
            if line.startswith('Epoch:'):
                try:
                    for m in range(line_index, -1, -1):
                        if ':' in lines[m] or '/' in lines[m]:
                            continue
                        else:
                            Methods.append(lines[m].strip())
                            break
                    self.processline(lines[line_index+2], lines[line_index+1], line, Methods, data_list, parameters)
                except IndexError:
                    # 如果出现索引超出范围的错误，打印错误信息并继续处理下一条记录
                    print(f"Skipping data due to index out of range at line {line_index}")
                    continue  # 继续下一次循环
        # 将数据列表转换为Pandas DataFrame
        self.data_frame = pd.DataFrame(data_list, columns=parameters)

    def processline(self, featureline, subfeature, line, Methods, data_list, parameters):
        row_data = {'Method': Methods[len(
            Methods)-1], 'Epoch': str(line.split(':')[-1]).strip()}
        if "tensor(" in featureline:
            featureline = featureline.replace("'", '"')
            featureline = re.sub(r"tensor\(([^)]+)\)", r"\1", featureline)
            features = featureline.split(';')
            for feature in features:
                if '{' in feature:
                    feature = json.loads(feature)
                    for key, value in feature.items():
                        if type(value) == dict:
                            for k, v in value.items():
                                row_data[key+'_'+k] = v
                        else:
                            row_data[key] = value
                else:
                    key, value = feature.split(':')
                    row_data[key.strip()] = value.strip()

        else:
            features = str(featureline).split(';')[0].split('\t')[:-1]
            for feature in features:
                if feature == 'Test' or feature in Methods:
                    continue
                print(feature)
                key, value = feature.split(':')
                row_data[key.strip()] = value.strip()
        
        subfeature = str(subfeature).split('Time')[1]
        match = re.search(r'\w+\.?\w+', subfeature)
        if match:
            # 提取匹配的部分
            key_string = match.group()
        key, value = 'Time',key_string
        print(key, value,"\n")
        row_data[key.strip()] = value.strip()

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
