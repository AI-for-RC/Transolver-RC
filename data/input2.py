import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re


def process_node_file(node_file_path):
    """处理NODE文件，提取弹性单元节点坐标、约束和荷载"""
    # 初始化弹性单元节点数据 [54, 9]
    # 9个参数: [x坐标, y坐标, z坐标, x约束, y约束, z约束, x荷载, y荷载, z荷载]
    elastic_nodes_data = np.zeros((54, 9), dtype=np.float32)

    # 弹性节点ID范围 (1837-1890)
    elastic_node_ids = list(range(1837, 1891))

    # 有特殊z方向荷载的节点ID
    special_load_nodes = [1838, 1839, 1843, 1846, 1849, 1852]

    with open(node_file_path, 'r') as file:
        for line in file:
            if line.startswith('NODE'):
                parts = line.split()
                if len(parts) >= 6:
                    node_id = int(parts[1])
                    if node_id in elastic_node_ids:
                        # 计算在弹性节点数组中的索引
                        index = node_id - 1837

                        # 提取坐标
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])

                        # 提取约束信息
                        constraint_code = int(parts[5])
                        x_constraint = (constraint_code // 100) % 10
                        y_constraint = (constraint_code // 10) % 10
                        z_constraint = constraint_code % 10

                        # 初始化荷载 (前6个荷载步使用第一个荷载步的数据)
                        x_load = 0.0
                        y_load = 0.0
                        z_load = -0.02 if node_id in special_load_nodes else 0.0

                        # 保存到弹性节点数据中
                        elastic_nodes_data[index] = [x, y, z, x_constraint, y_constraint, z_constraint, x_load, y_load,
                                                     z_load]

    return elastic_nodes_data


def generate_load_steps(first_step_data):
    """生成49个荷载步的数据"""
    # 复制第一荷载步的数据到所有49个荷载步
    all_steps_data = np.repeat(np.expand_dims(first_step_data, axis=2), 49, axis=2)

    # 修改特殊节点的z方向荷载
    special_indices = [node_id - 1837 for node_id in [1838, 1839, 1843, 1846, 1849, 1852]]

    for step in range(1, 49):
        # 计算当前荷载步的z方向荷载 (-0.02, -0.04, ..., -0.98)
        z_load = -0.02 * (step + 1)

        # 对特殊节点应用当前荷载步的z方向荷载
        for idx in special_indices:
            all_steps_data[idx, 6, step] = 0.0  # x荷载保持0
            all_steps_data[idx, 7, step] = 0.0  # y荷载保持0
            all_steps_data[idx, 8, step] = z_load  # z荷载按步长递减

    return all_steps_data


def process_beam_folder(folder_path):
    """处理单个梁文件夹，返回弹性节点的所有荷载步数据"""
    node_file = os.path.join(folder_path, 'NODE')

    # 处理NODE文件获取第一个荷载步的数据
    first_step_data = process_node_file(node_file)

    # 生成所有49个荷载步的数据
    all_load_steps = generate_load_steps(first_step_data)

    return all_load_steps


def main():
    """主函数，处理所有梁文件夹并构建四维矩阵"""
    # 设置根目录，这里假设500个文件夹都在当前目录下
    root_directory = r"C:\Users\ranxiaoyang\Desktop\1-500\new"

    # 获取所有包含NODE文件的文件夹
    beam_folders = []
    for item in os.listdir(root_directory):
        item_path = os.path.join(root_directory, item)
        if os.path.isdir(item_path):
            node_path = os.path.join(item_path, 'NODE')
            if os.path.exists(node_path):
                beam_folders.append(item_path)

    print(f"找到 {len(beam_folders)} 个包含NODE文件的文件夹")

    # 初始化四维矩阵 [500, 54, 9, 49]
    matrix_shape = (len(beam_folders), 54, 9, 49)
    data_matrix = np.zeros(matrix_shape, dtype=np.float32)

    # 处理每个梁文件夹
    for i, folder_path in enumerate(tqdm(beam_folders, desc="处理梁文件夹")):
        # 获取当前梁的弹性节点在所有荷载步的数据 [54, 9, 49]
        beam_data = process_beam_folder(folder_path)

        # 保存到四维矩阵中
        data_matrix[i] = beam_data

    # 保存矩阵到文件
    output_file = "input2.npy"
    np.save(output_file, data_matrix)

    # 验证矩阵维度
    actual_shape = data_matrix.shape
    expected_shape = (500, 54, 9, 49)

    if actual_shape == expected_shape:
        print(f"成功保存四维矩阵到 {output_file}，维度为 {actual_shape}")
    else:
        print(f"矩阵维度不匹配！预期 {expected_shape}，实际 {actual_shape}")
        if len(beam_folders) != 500:
            print(f"提示：找到 {len(beam_folders)} 个梁文件夹，而不是500个")


if __name__ == "__main__":
    main()