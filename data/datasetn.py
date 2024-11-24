import os
import numpy as np
from util.selected_ob_metrics import selected_ob_metrics

case_split = 5 # 把一个case切成多个case，增加数据量 单节点5

# 读取单种类异常的数据文件夹，并加载标签
# 目前使用多个节点，同时返回：
#
def read_case_node_n(datapath:str,anomaly_case:str, node_num):
    # 首先获取当前文件夹中有几个case
    labels = [] # 4个Node，分开存   [case1labels,case2label] case1label: {node1:[labels],node1:[labels],node1:[labels]}
    data = []  # 4个Node，分开存    data = [[node1_data, node2_data, node3_data] for case1]

    anomaly_case = os.path.join(datapath, anomaly_case)
    with open(os.path.join(anomaly_case, "cases.txt"), "r") as f:
        case_number = int(f.readline())

    # 逐个读取所有case
    for case_number in range(1, case_number + 1):
        current_case_dir = os.path.join(anomaly_case, f"case{case_number}")

        case_labels = {}
        # 读取节点label
        label_file = os.path.join(current_case_dir, "label.txt")
        with open(label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                [node, label] = line.strip().split(".")
                if node not in case_labels:
                    case_labels[node] = [label]
                else:
                    case_labels[node].append(label)


        case_data_split = [[] for _ in range(case_split)]
        # 读取dstat与ob数据，三个label写死
        for node_number in range(1, node_num + 1):
            dstat_report = os.path.join(current_case_dir, f"dstat_report{node_number}.csv")
            ob_report = os.path.join(current_case_dir, f"ob_report{node_number}.csv")
            dstat_data = read_dstat_csv(dstat_report)
            ob_data = read_ob_report_csv(ob_report)

            if len(ob_data) % case_split != 0:
                raise Exception("can not split case")
            st_idx = 0
            idx_number = len(ob_data) // case_split

            for case_idx in range(case_split):
                node_data = []

                for i in range(st_idx,st_idx+idx_number):
                    dstat_data[i].extend(ob_data[i])
                    node_data.append(dstat_data[i])

                case_data_split[case_idx].append(node_data)
                st_idx += idx_number


        for case_data in case_data_split:
            data.append(case_data)
            labels.append(case_labels)  # labels同样重复

    return data, labels


def read_dstat_csv(file):
    data = []
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines[7:]:
            number = [float(x) for x in line.strip().split(",")]
            data.append(number)
    return data

def read_ob_report_csv(file):
    data = []
    with open(file, "r") as f:
        lines = f.readlines()
        first_line_metric = list(map(float,lines[1].strip().split(",")))
        first_numer = []
        for (name, idx, is_accum) in selected_ob_metrics:
            if is_accum:
                first_numer.append(0.0)
            else:
                first_numer.append(first_line_metric[idx])
        data.append(first_numer)

        for line in lines[2:]:
            metric_lst = line.strip().split(",")
            number = []
            for (name, idx, is_accum) in selected_ob_metrics:
                if is_accum:
                    number.append(float(metric_lst[idx])-first_line_metric[idx])
                else:
                    number.append(float(metric_lst[idx]))

            data.append(number)
    return data



def prepare_data_for_distdiagnosis_node_n(datapath,datasets, classes, node_num):
    all_training_data = []
    all_training_labels = []
    all_node_metric = []
    for dataset in datasets:
        data, labels = read_case_node_n(datapath,dataset,node_num)
        # distdiagnosis训练数据，分4个node，每个case的data直接展平，打上

        training_data = []
        training_labels = []
        node_metric = []

        # 1、训练distdiagnosis中单节点层级诊断器的数据
        for i in range(len(data)):
            # 三个node的数据和label拆开，label是一个多分类标签，DistDiagnosis内部再拆开多标签训练1对多分类模型
            data_per_case = []
            label_per_case = []

            for node_idx in range(node_num):
                tmp_data = np.array(data[i][node_idx])
                tmp_data = tmp_data.flatten()
                tmp_label = [0] * len(classes)
                for root_cause in labels[i][f'node{node_idx+1}']:
                    if root_cause in classes:
                        root_cause_idx = classes.index(root_cause)
                        tmp_label[root_cause_idx] = 1

                data_per_case.append(tmp_data)
                label_per_case.append(tmp_label)

            training_data.append(data_per_case)
            training_labels.append(label_per_case)

            # 2、进行图边权计算的节点指标数据
            node_metric.append(data[i])

        # 加到总的里面
        all_training_data.extend(training_data)
        all_training_labels.extend(training_labels)
        all_node_metric.extend(node_metric)

    pass

    return all_training_labels, all_training_data, all_node_metric


def prepare_data_for_baseline_node_n(datapath,datasets, classes,node_num):

    all_training_labels = []
    all_training_data = []
    for dataset in datasets:
        data, labels = read_case_node_n(datapath,dataset,node_num)
        # xgboost等的训练数据，直接4个node的数据flatten后拼在一起
        # 标签也拼在一起

        training_data = []
        training_labels = []

        #
        for i in range(len(data)):

            data_per_case = []
            label_per_case = []

            for node_idx in range(node_num):
                tmp_data = np.array(data[i][node_idx])
                tmp_data = tmp_data.flatten()
                tmp_label = [0] * len(classes)
                for root_cause in labels[i][f'node{node_idx + 1}']:
                    if root_cause in classes:
                        root_cause_idx = classes.index(root_cause)
                        tmp_label[root_cause_idx] = 1

                data_per_case.append(tmp_data)
                label_per_case.extend(tmp_label)

            training_data.append(np.concatenate(tuple(data_per_case), axis=0))
            training_labels.append(label_per_case)

        all_training_data.extend(training_data)
        all_training_labels.extend(training_labels)

    return all_training_labels, all_training_data


if __name__ == "__main__":
    pass