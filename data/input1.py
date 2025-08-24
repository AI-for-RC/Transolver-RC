import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re


def process_node_file(node_file_path):
    """处理NODE文件，提取节点坐标"""
    node_coordinates = np.zeros((1836, 3))  # 初始化节点坐标数组

    with open(node_file_path, 'r') as file:
        for line in file:
            if line.startswith('NODE'):
                parts = line.split()
                if len(parts) >= 6:
                    node_id = int(parts[1])
                    if 1 <= node_id <= 1836:
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])
                        node_coordinates[node_id - 1] = [x, y, z]  # 节点ID从1开始，数组索引从0开始

    return node_coordinates


def process_elem_file(elem_file_path, node_rebar_ratios):
    """处理ELEM文件，提取配筋率并映射到节点"""
    # 初始化配筋率数组，每个节点在三个方向上的配筋率

    with open(elem_file_path, 'r') as file:
        lines = file.readlines()
        line_index = 0

        while line_index < len(lines):
            line = lines[line_index].strip()
            if line.startswith('ELEM'):
                # 解析单元行
                parts = line.split()
                if len(parts) >= 10:
                    # 获取单元包含的节点ID
                    elem_nodes = [int(parts[i]) for i in range(2, 10)]  # 第3到第10列是节点ID

                    # 读取接下来的3行，获取三个方向的配筋率
                    rebar_ratios = []
                    line_index += 1
                    if lines[line_index + 1].strip().startswith('ELEM'):
                        return node_rebar_ratios
                    for _ in range(3):
                        line_index += 1
                        if line_index < len(lines):
                            rebar_line = lines[line_index].strip()
                            rebar_parts = rebar_line.split()
                            if len(rebar_parts) >= 6:
                                # 第6列是配筋率
                                rebar_ratios.append(float(rebar_parts[5]))
                            else:
                                rebar_ratios.append(0.0)
                        else:
                            rebar_ratios.append(0.0)

                    # 将配筋率映射到节点
                    for node_id in elem_nodes:
                        if 1 <= node_id <= 1836:
                            # 对每个方向的配筋率赋值
                            for direction in range(3):
                                node_rebar_ratios[node_id - 1, direction] = rebar_ratios[direction]

            line_index += 1

    return node_rebar_ratios


def process_beam_folder(folder_path, node_rebar_ratios):
    """处理单个梁文件夹，返回节点的完整特征"""
    node_file = os.path.join(folder_path, 'INPUT.DAT')
    elem_file = os.path.join(folder_path, 'INPUT.DAT')

    # 处理NODE文件获取坐标
    coordinates = process_node_file(node_file)

    # 处理ELEM文件获取配筋率
    rebar_ratios = process_elem_file(elem_file, node_rebar_ratios)

    # 组合坐标和配筋率，形成 [1836, 6] 的特征矩阵
    # 6个参数: [x坐标, y坐标, z坐标, x方向配筋率, y方向配筋率, z方向配筋率]
    node_features = np.hstack((coordinates, rebar_ratios))

    return node_features


def main():
    """主函数，处理所有梁文件夹并构建四维矩阵"""
    # 设置根目录，500个文件夹都在当前目录下
    root_directory = r"D:\Transolver-RC\data\1-500\new"

    # 获取所有包含NODE和ELEM文件的文件夹
    beam_folders = []
    i = 0
    for item in os.listdir(root_directory):
        item_path = os.path.join(root_directory, item)
        beam_folders.append(item_path)
        i += 1
        if i >= 500:
            break

    print(f"找到 {len(beam_folders)} 个包含NODE和ELEM文件的文件夹")

    # 初始化四维矩阵 [500, 1836, 6, 49]
    matrix_shape = (len(beam_folders), 1836, 6, 49)
    data_matrix = np.zeros(matrix_shape, dtype=np.float32)

    node_rebar_ratios = np.zeros((1836, 3))  # [节点ID, 方向(x=0,y=1,z=2)]

    # 处理每个梁文件夹
    for i, folder_path in enumerate(tqdm(beam_folders, desc="处理梁文件夹")):
        # 获取当前梁的节点特征 [1836, 6]
        node_features = process_beam_folder(folder_path, node_rebar_ratios)

        for node_id in range(307, 613):
            for direction in range(3):
                avg_ratio = (node_features[0, direction + 3] + node_features[612, direction + 3]) / 2
                node_features[node_id - 1, direction + 3] = avg_ratio

        # 将特征复制到所有49个荷载步
        data_matrix[i, :, :, :] = np.expand_dims(node_features, axis=2).repeat(49, axis=2)


    # 保存矩阵到文件
    output_file = "newinput1.npy"
    np.save(output_file, data_matrix)

    # 验证矩阵维度
    actual_shape = data_matrix.shape
    expected_shape = (500, 1836, 6, 49)

    if actual_shape == expected_shape:
        print(f"成功保存四维矩阵到 {output_file}，维度为 {actual_shape}")
    else:
        print(f"矩阵维度不匹配！预期 {expected_shape}，实际 {actual_shape}")
        if len(beam_folders) != 500:
            print(f"提示：找到 {len(beam_folders)} 个梁文件夹，而不是500个")


if __name__ == "__main__":
    main()