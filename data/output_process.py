import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

all_outputs = []

for folder_idx in range(1, 501):
    file_path = os.path.join(base_dir, str(folder_idx), 'INPUT-MECH.nod')
    
    num_load_steps = 49
    nodes_per_step = 1890
    
    disp = []
    
    with open(file_path, 'r') as f:
        for step_idx in range(num_load_steps):
            line1 = f.readline()
            if not line1.strip():
                break

            line2 = f.readline()
            if not line2.strip():
                break

            current_step_nodes_data = []
            for node_idx in range(nodes_per_step):
                node_line = f.readline().strip()
                if not node_line:
                    break 
                
                values = node_line.split()
                
                x_disp = float(values[1])
                y_disp = float(values[2])
                z_disp = float(values[3])
                current_step_nodes_data.append([x_disp, y_disp, z_disp])
            
            disp.append(np.array(current_step_nodes_data))
    
    output_disp = np.array(disp)
    output_disp = np.transpose(output_disp, (1, 2, 0))
    
    file_path = os.path.join(base_dir, str(folder_idx), 'INPUT-MECH.fld')
    
    num_load_steps = 49
    elems_per_step = 1280
    
    gauss_data = []
    
    with open(file_path, 'r') as f:
        for step_idx in range(num_load_steps):
            line1 = f.readline()
            if not line1.strip():
                break

            line2 = f.readline()
            if not line2.strip():
                break

            current_step_elems_data = []
            for elem in range(elems_per_step):
                current_step_gauss_data = []
                for gauss in range(8):
                    node_line = f.readline().strip()
                    if not node_line:
                        break 
                    values = node_line.split()
                    xx_stress = float(values[5])
                    yy_stress = float(values[6])
                    zz_stress = float(values[7])
                    xy_stress = float(values[8])
                    xz_stress = float(values[9])
                    yz_stress = float(values[10])
                    xx_strain = float(values[11])
                    yy_strain = float(values[12])
                    zz_strain = float(values[13])
                    xy_strain = float(values[14])
                    xz_strain = float(values[15])
                    yz_strain = float(values[16])
                    current_step_gauss_data.append([xx_stress, yy_stress, zz_stress, xy_stress, xz_stress, yz_stress, xx_strain, yy_strain, zz_strain, xy_strain, xz_strain, yz_strain])
                current_step_elems_data.append(np.array(current_step_gauss_data))
            gauss_data.append(np.array(current_step_elems_data))
    
    output_gauss = np.array(gauss_data)
    output_gauss = np.transpose(output_gauss, (1, 2, 3, 0))
    
    node_gauss = []
    for node_idx in range(1, nodes_per_step + 1):
        x_index = (node_idx - 1) // 306 + 1
        x_rem = (node_idx - 1) % 306
        y_index = x_rem // 51 + 1
        y_rem = x_rem % 51
        z_index = y_rem + 1

        x_1 = x_index - 1
        y_1 = y_index
        z_1 = z_index
        if x_1 < 1 or x_1 > 5 or y_1 < 1 or y_1 > 5 or z_1 < 1 or z_1 > 50:
            elem1 = 0
        else:
            elem1 = (x_1 - 1) * 250 + (y_1 - 1) * 50 + z_1

        x_2 = x_index - 1
        y_2 = y_index
        z_2 = z_index - 1
        if x_2 < 1 or x_2 > 5 or y_2 < 1 or y_2 > 5 or z_2 < 1 or z_2 > 50:
            elem2 = 0
        else:
            elem2 = (x_2 - 1) * 250 + (y_2 - 1) * 50 + z_2

        x_3 = x_index - 1
        y_3 = y_index - 1
        z_3 = z_index
        if x_3 < 1 or x_3 > 5 or y_3 < 1 or y_3 > 5 or z_3 < 1 or z_3 > 50:
            elem3 = 0
        else:
            elem3 = (x_3 - 1) * 250 + (y_3 - 1) * 50 + z_3

        x_4 = x_index - 1
        y_4 = y_index - 1
        z_4 = z_index - 1
        if x_4 < 1 or x_4 > 5 or y_4 < 1 or y_4 > 5 or z_4 < 1 or z_4 > 50:
            elem4 = 0
        else:
            elem4 = (x_4 - 1) * 250 + (y_4 - 1) * 50 + z_4

        x_5 = x_index
        y_5 = y_index
        z_5 = z_index
        if x_5 < 1 or x_5 > 5 or y_5 < 1 or y_5 > 5 or z_5 < 1 or z_5 > 50:
            elem5 = 0
        else:
            elem5 = (x_5 - 1) * 250 + (y_5 - 1) * 50 + z_5

        x_6 = x_index
        y_6 = y_index
        z_6 = z_index - 1
        if x_6 < 1 or x_6 > 5 or y_6 < 1 or y_6 > 5 or z_6 < 1 or z_6 > 50:
            elem6 = 0
        else:
            elem6 = (x_6 - 1) * 250 + (y_6 - 1) * 50 + z_6

        x_7 = x_index
        y_7 = y_index - 1
        z_7 = z_index
        if x_7 < 1 or x_7 > 5 or y_7 < 1 or y_7 > 5 or z_7 < 1 or z_7 > 50:
            elem7 = 0
        else:
            elem7 = (x_7 - 1) * 250 + (y_7 - 1) * 50 + z_7

        x_8 = x_index
        y_8 = y_index - 1
        z_8 = z_index - 1
        if x_8 < 1 or x_8 > 5 or y_8 < 1 or y_8 > 5 or z_8 < 1 or z_8 > 50:
            elem8 = 0
        else:
            elem8 = (x_8 - 1) * 250 + (y_8 - 1) * 50 + z_8

        node_gauss.append([elem1, elem2, elem3, elem4, elem5, elem6, elem7, elem8])
    
    node_gauss = np.array(node_gauss)
    
    num_nodes, num_disp_dims, num_steps = output_disp.shape
    num_gauss_per_node = node_gauss.shape[1]
    num_gauss_points = output_gauss.shape[2]
    
    output = np.zeros((num_nodes, 3 + num_gauss_per_node * num_gauss_points, num_steps))
    output[:, :3, :] = output_disp
    
    for node_idx in range(num_nodes):
        for gauss_idx in range(num_gauss_per_node):
            element_number = node_gauss[node_idx, gauss_idx]
            if element_number == 0:
                output[node_idx, 3 + gauss_idx * num_gauss_points:3 + (gauss_idx + 1) * num_gauss_points, :] = 0
            else:
                output[node_idx, 3 + gauss_idx * num_gauss_points:3 + (gauss_idx + 1) * num_gauss_points, :] = output_gauss[element_number - 1, gauss_idx, :, :]
    
    all_outputs.append(output)
    print(f"folder {folder_idx} done")

output = np.array(all_outputs)
print(output.shape)
file_name = os.path.join(base_dir, 'output.npy')
np.save(file_name, output)

