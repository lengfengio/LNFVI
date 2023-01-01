import os
import numpy as np
def gen_flow_refine_test_mask_list(flow_root, output_txt_path):
    output_txt = open(output_txt_path, 'w')
    flow_list = [x for x in os.listdir(flow_root) if 'flo' in x]
    flow_no_list = [int(x[:5]) for x in flow_list]
    flow_start_no = min(flow_no_list)
    flow_num = len(flow_list) // 2
    for i in range(flow_start_no - 5, flow_start_no + flow_num - 4):
        gt_flow_no = [0, 0]
        f_flow_no = []
        for k in range(11):
            flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
            f_flow_no.append(int(flow_no))
            output_txt.write('%05d.flo' % flow_no)
            if k == 5:
                gt_flow_no[0] = flow_no
            output_txt.write(' ')
        r_flow_no = []
        for k in range(11):
            flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num)
            r_flow_no.append(int(flow_no))
            if k == 5:
                gt_flow_no[1] = flow_no
            output_txt.write('%05d.rflo' % flow_no)
            output_txt.write(' ')
        for k in range(11):
            output_txt.write('%05d.png' % f_flow_no[k])
            output_txt.write(' ')
        for k in range(11):
            output_txt.write('%05d.png' % r_flow_no[k])
            output_txt.write(' ')
        output_path = ','.join(['%05d.flo' % gt_flow_no[0],'%05d.rflo' % gt_flow_no[1]])
        output_txt.write(output_path)
        output_txt.write(' ')
        output_txt.write(str(0))
        output_txt.write('\n')
    output_txt.close()
def gen_flow_refine_test_mask_list_Nonlocal(flow_root, output_txt_path):
    output_txt = open(output_txt_path, 'w')
    flow_list = [x for x in os.listdir(flow_root) if 'flo' in x]
    flow_no_list = [int(x[:5]) for x in flow_list]
    flow_start_no = min(flow_no_list)
    flow_num = len(flow_list) // 2
    for i in range(flow_start_no - 5, flow_start_no + flow_num - 4):
        gt_flow_no = [0, 0]
        f_flow_no = []
        r_flow_no = []
        if i != flow_start_no + flow_num - 25 and i != flow_start_no + flow_num - 5:
            index = 0
            for k in range(11):
                flow_no = np.clip(i + index, a_min=flow_start_no, a_max=flow_start_no + flow_num)
                if flow_no >= flow_start_no + flow_num:
                    flow_no = i + index - flow_num
                f_flow_no.append(int(flow_no))
                output_txt.write('%05d.flo' % flow_no)
                if k == 5:
                    gt_flow_no[0] = flow_no
                output_txt.write(' ')
                index = index + 5
            index = 0
            for k in range(11):
                flow_no = np.clip(i + index, a_min=flow_start_no, a_max=flow_start_no + flow_num)
                if flow_no >= flow_start_no + flow_num:
                    flow_no = i + index - flow_num
                r_flow_no.append(int(flow_no))
                if k == 5:
                    gt_flow_no[1] = flow_no
                output_txt.write('%05d.rflo' % flow_no)
                output_txt.write(' ')
                index = index + 5
        elif i == flow_start_no + flow_num - 25:
            index = 0
            for k in range(11):
                flow_no = np.clip(i + index, a_min=flow_start_no, a_max=flow_start_no + flow_num)
                if flow_no == int(flow_num):
                    flow_no = 0
                f_flow_no.append(int(flow_no))
                if k == 5:
                    gt_flow_no[0] = flow_no
                output_txt.write('%05d.flo' % flow_no)
                output_txt.write(' ')
                index = index + 5
            index = 0
            for k in range(11):
                flow_no = np.clip(i + index, a_min=flow_start_no, a_max=flow_start_no + flow_num)
                r_flow_no.append(int(flow_no))
                if k == 5:
                    gt_flow_no[1] = flow_no
                output_txt.write('%05d.rflo' % flow_no)
                output_txt.write(' ')
                index = index + 5
        elif i == flow_start_no + flow_num - 5:
            print(i)
            index = 0
            for k in range(11):
                flow_no = 0
                f_flow_no.append(int(flow_no))
                if k == 5:
                    gt_flow_no[0] = flow_no
                output_txt.write('%05d.flo' % flow_no)
                output_txt.write(' ')
                index = index + 1
            index = 0
            for k in range(11):
                flow_no = 0
                r_flow_no.append(int(flow_no))
                if k == 5:
                    gt_flow_no[1] = flow_no
                output_txt.write('%05d.rflo' % flow_no)
                output_txt.write(' ')
                index = index + 1
        for k in range(11):
            output_txt.write('%05d.png' % f_flow_no[k])
            output_txt.write(' ')
        for k in range(11):
            output_txt.write('%05d.png' % r_flow_no[k])
            output_txt.write(' ')
        output_path = ','.join(['%05d.flo' % gt_flow_no[0],'%05d.rflo' % gt_flow_no[1]])
        output_txt.write(output_path)
        output_txt.write(' ')
        output_txt.write(str(0))
        output_txt.write('\n')
    output_txt.close()