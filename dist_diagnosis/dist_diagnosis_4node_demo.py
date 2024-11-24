
from data.dataset import prepare_data_for_baseline, prepare_data_for_distdiagnosis
from data.dataset4 import prepare_data_for_distdiagnosis_4node, prepare_data_for_baseline_4node

import random

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np

from xgboost import XGBClassifier
from page_rank import WPR4Node
from scipy.stats import pearsonr

class DistDiagnosis_4node():
    def __init__(self, num_classifier, classes_for_each_node):
        self.num_classifier = num_classifier
        self.clf_list = []
        for _ in range(num_classifier):

            self.clf_list.append(XGBClassifier(objective='binary:logistic'))
            #self.clf_list.append(RandomForestClassifier())
        self.classes = classes_for_each_node

    def train(self, x_train, y_train): # X_train和y_train里面是分多个node的
        train_x = self.decompose_train_x(x_train)
        train_y = self.decompose_train_y(y_train)


        for i in range(self.num_classifier): # 每一个类别训一个节点层级分类器
            current_clf = self.clf_list[i]
            current_labels = train_y[:, i].reshape(-1, 1)
            current_clf.fit(train_x, current_labels)
        pass


    def predict(self, X_test, node_metric):

        res = []
        for i in range(len(X_test)):
            # 计算当前case中节点异常分
            scores = self.calculate_score_for_node(node_metric[i])

            # 各节点预测
            tmp_label = []
            nodes_x = X_test[i]
            for clf in self.clf_list:
                pred_node_one_class = clf.predict(nodes_x)
                tmp_label.append(pred_node_one_class)
            line = []
            for i in range(len(tmp_label[0])):
                for j in range(len(tmp_label)):
                    line.append(tmp_label[j][i] * scores[i]) # 加权最终得分
            res.append(line)

        # 构成root_cause元组
        root_cause = []
        for p_case in res:
            ordered_root_cause = []
            for i in range(4):
                each_node_class = self.classes

                for j in range(len(each_node_class)):
                    # (异常名，得分，ylabel中的下标)
                    idx = i * len(each_node_class) + j
                    ordered_root_cause.append(("node" + str(i+1) + "." + each_node_class[j], p_case[idx], idx))
            random.shuffle(ordered_root_cause)
            ordered_root_cause.sort(key=lambda x: x[1], reverse=True)
            root_cause.append(ordered_root_cause)
        return root_cause


    def decompose_train_x(self, composed_train_x):
        train_x = []
        for case in composed_train_x:
            for node_x in case:
                train_x.append(node_x)
        return np.array(train_x)

    def decompose_train_y(self, composed_train_y):
        train_y = []
        for case in composed_train_y:
            for node_label in case:
                train_y.append(node_label)
        return np.array(train_y)

    def calculate_score_for_node(self, node_metric_one_case):
        w01 = self.calculate_rank(node_metric_one_case[0], node_metric_one_case[1])
        w02 = self.calculate_rank(node_metric_one_case[0], node_metric_one_case[2])
        w03 = self.calculate_rank(node_metric_one_case[0], node_metric_one_case[3])
        w12 = self.calculate_rank(node_metric_one_case[1], node_metric_one_case[2])
        w13 = self.calculate_rank(node_metric_one_case[1], node_metric_one_case[3])
        w23 = self.calculate_rank(node_metric_one_case[2], node_metric_one_case[3])

        w = [w01, w02, w03, w12, w13, w23]
        pr = WPR4Node(w)
        score = pr.page_rank(10)
        return score

    def calculate_rank(self, node_x, node_y):
        x_array = np.array(node_x)
        y_array = np.array(node_y)
        rvalue = []
        for i in range(len(x_array[0])):
            x_a = x_array[:,i]
            y_a = y_array[:,i]
            coef, p_value = pearsonr(x_a, y_a)
            if coef >= -1 or coef <= 1:
                rvalue.append(abs(coef))


        res = sum(rvalue) / len(rvalue)
        return res


def ac_at_k(pred_root_cause, y_label, k):

    total_ac = []
    for i in range(len(pred_root_cause)):
        cnt = 0
        total_number = min(k, sum(y_label[i])) # 有几个发生的根因root cause
        if total_number == 0:
            continue
        current_pred = pred_root_cause[i]
        for (name, score, idx) in current_pred[0:k]:
            if y_label[i][idx] == 1:
                cnt += 1
        ac = cnt/total_number
        total_ac.append(ac)

    return sum(total_ac)/len(total_ac)


def get_split_idx(total_number, test_ratio=0.1):
    test_num = int(total_number * test_ratio)

    total_idx = list(range(total_number))
    test_idx = random.sample(total_idx, test_num)
    train_idx = [x for x in total_idx if x not in test_idx]
    return train_idx, test_idx

def split_training_test(total_x, total_label, train_idx, test_idx):
    train_x = [total_x[i] for i in train_idx]
    test_x = [total_x[i] for i in test_idx]

    train_y = [total_label[i] for i in train_idx]
    test_y = [total_label[i] for i in test_idx]

    return train_x, train_y, test_x, test_y

def get_test_node_metric(node_metric, test_idx):
    test_node_metric = []
    for idx in test_idx:
        test_node_metric.append(node_metric[idx])
    return test_node_metric

def distdiagnosis_ac_at_k(pred_root_cause, composed_y_label, k):

    total_ac = []
    for i in range(len(pred_root_cause)):
        cnt = 0
        root_cause_num = 0
        for node in composed_y_label[i]:
            root_cause_num += sum(node)

        total_number = min(k, root_cause_num) # 有几个发生的根因root cause
        if total_number == 0:
            continue
        current_pred = pred_root_cause[i]
        for (name, score, idx) in current_pred[0:k]:
            decomposed_label = []
            for lst in composed_y_label[i]:
                decomposed_label.extend(lst)
            if decomposed_label[idx] == 1:
                cnt += 1
        ac = cnt/total_number
        total_ac.append(ac)

    return sum(total_ac)/len(total_ac)

def get_split_test_x(all_x, train_idx, num_classes):
    # 切出不同class的单独test_x
    # 由于每个class的数据在all_x里是连续的，即可判断
    split_test_x = [[] for _ in range(num_classes)]

    length_per_class = int(len(all_x) / num_classes)

    for idx in train_idx:
        class_id = idx // length_per_class
        if class_id >= num_classes:
            class_id -= 1
        split_test_x[class_id].append(all_x[idx])
    return split_test_x


def evaluate_distdiagnosis_3_for_4():
    read_dataset = ["cpu_saturation", #"io_saturation", "net_saturation","buffer",
                    "cpu_2_saturation", #"io_2_saturation", "net_2_saturation", "buffer_2",
                    "io_and_cpu_saturation",#"cpu_and_net_saturation","io_and_net_saturation","buffer_and_io_saturation",
                    "cpu_saturation_and_load"]#,"io_saturation_and_load","net_saturation_and_toomanyindex","buffer_and_toomanyindex_2"]
                    # "normal"]
    labels_to_evaluate = ["cpu", "io", "buffer", "net", "load"]


    #random.seed(42)
    random.seed(1314)
    # 3节点数据做训练
    training_labels, training_data, node_metric = prepare_data_for_distdiagnosis(read_dataset, labels_to_evaluate)

    print("Total data: ",len(training_labels))
    train_idx, test_idx = get_split_idx(len(training_labels), 0) # 用作训练

    train_x, train_y, test_x, test_y = split_training_test(np.array(training_data), np.array(training_labels),
                                                           train_idx, test_idx)


    om = DistDiagnosis_4node(num_classifier=len(labels_to_evaluate), classes_for_each_node=labels_to_evaluate)
    om.train(train_x, train_y)

    # 四节点做测试
    read_dataset = ["cpu_saturation","cpu_2_saturation","io_and_cpu_saturation","cpu_saturation_and_load"]
    test_labels, test_data, test_node_metric = prepare_data_for_distdiagnosis_4node(read_dataset, labels_to_evaluate)
    root_cause = om.predict(test_data, test_node_metric)

    om_ac_1 = distdiagnosis_ac_at_k(root_cause, test_labels, 1)
    om_ac_2 = distdiagnosis_ac_at_k(root_cause, test_labels, 2)
    om_ac_3 = distdiagnosis_ac_at_k(root_cause, test_labels, 3)
    om_ac_4 = distdiagnosis_ac_at_k(root_cause, test_labels, 4)
    om_ac_5 = distdiagnosis_ac_at_k(root_cause, test_labels, 5)
    print("overall ac@1", om_ac_1)
    print("overall ac@2", om_ac_2)
    print("overall ac@3", om_ac_3)
    print("overall ac@4", om_ac_4)
    print("overall ac@5", om_ac_5)

    split_test_x = get_split_test_x(test_data, list(range(len(test_data))), len(read_dataset))
    split_test_node_metric = get_split_test_x(test_node_metric, list(range(len(test_data))), len(read_dataset))
    split_test_y = get_split_test_x(test_labels, list(range(len(test_data))), len(read_dataset))
    for i in range(len(split_test_x)):
        root_cause = om.predict(split_test_x[i],split_test_node_metric[i])
        om_ac_1 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 1)
        om_ac_2 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 2)
        om_ac_3 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 3)
        om_ac_4 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 4)
        om_ac_5 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 5)
        print(f"class {i} ac@1", om_ac_1)
        print(f"class {i} ac@2", om_ac_2)
        print(f"class {i} ac@3", om_ac_3)
        print(f"class {i} ac@4", om_ac_4)
        print(f"class {i} ac@5", om_ac_5)

    return test_idx, train_idx


def evaluate_distdiagnosis_4_for_4():

    read_dataset = ["cpu_saturation","cpu_2_saturation","io_and_cpu_saturation","cpu_saturation_and_load"]
    labels_to_evaluate = ["cpu", "io", "buffer", "net", "load"]

    #random.seed(42)
    random.seed(1314)
    training_labels, training_data, node_metric = prepare_data_for_distdiagnosis_4node(read_dataset, labels_to_evaluate)
    print("Total data: ",len(training_labels))
    train_idx, test_idx = get_split_idx(len(training_labels), 0.50)

    train_x, train_y, test_x, test_y = split_training_test(np.array(training_data), np.array(training_labels),
                                                           train_idx, test_idx)

    test_node_metric = get_test_node_metric(node_metric, test_idx)
    om = DistDiagnosis_4node(num_classifier=len(labels_to_evaluate), classes_for_each_node=labels_to_evaluate)
    om.train(train_x, train_y)
    root_cause = om.predict(test_x, test_node_metric)

    om_ac_1 = distdiagnosis_ac_at_k(root_cause, test_y, 1)
    om_ac_2 = distdiagnosis_ac_at_k(root_cause, test_y, 2)
    om_ac_3 = distdiagnosis_ac_at_k(root_cause, test_y, 3)
    om_ac_4 = distdiagnosis_ac_at_k(root_cause, test_y, 4)
    om_ac_5 = distdiagnosis_ac_at_k(root_cause, test_y, 5)
    print("overall ac@1", om_ac_1)
    print("overall ac@2", om_ac_2)
    print("overall ac@3", om_ac_3)
    print("overall ac@4", om_ac_4)
    print("overall ac@5", om_ac_5)

    split_test_x = get_split_test_x(training_data, test_idx, len(read_dataset))
    split_test_node_metric = get_split_test_x(node_metric, test_idx, len(read_dataset))
    split_test_y = get_split_test_x(training_labels, test_idx, len(read_dataset))
    for i in range(len(split_test_x)):
        root_cause = om.predict(split_test_x[i],split_test_node_metric[i])
        om_ac_1 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 1)
        om_ac_2 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 2)
        om_ac_3 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 3)
        om_ac_4 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 4)
        om_ac_5 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 5)
        print(f"class {i} ac@1", om_ac_1)
        print(f"class {i} ac@2", om_ac_2)
        print(f"class {i} ac@3", om_ac_3)
        print(f"class {i} ac@4", om_ac_4)
        print(f"class {i} ac@5", om_ac_5)

    return test_idx, train_idx

def evaluate_distdiagnosis_3_and_4_for_4():
    read_dataset = ["cpu_2_saturation", "io_2_saturation", "buffer_2", "net_2_saturation","load","normal"]
    labels_to_evaluate = ["cpu", "io", "buffer", "net", "load"]

    #random.seed(42)
    random.seed(1314)
    # 3节点数据做训练
    training_labels, training_data, node_metric = prepare_data_for_distdiagnosis(read_dataset, labels_to_evaluate)

    print("Total data: ",len(training_labels))
    train_idx, test_idx = get_split_idx(len(training_labels), 0) # 少用点数据训，不然效果太高

    train_x_3node, train_y_3node, test_x, test_y = split_training_test(np.array(training_data), np.array(training_labels),
                                                           train_idx, test_idx)


    read_dataset = ["cpu_saturation", "cpu_2_saturation", "io_and_cpu_saturation", "cpu_saturation_and_load"]
    # 四节点做测试
    test_labels_4node, test_data_4node, test_node_metric_4node = prepare_data_for_distdiagnosis_4node(
        ["cpu_saturation", "cpu_2_saturation", "io_and_cpu_saturation", "cpu_saturation_and_load"], labels_to_evaluate)
    train_idx, test_idx = get_split_idx(len(test_labels_4node), 0.5)
    train_x_4node, train_y_4node, test_x_4node, test_y_4node = split_training_test(np.array(test_data_4node),
                                                                       np.array(test_labels_4node),
                                                                       train_idx, test_idx)
    train_x = []
    train_x.extend(train_x_3node)
    train_x.extend(train_x_4node)
    train_y = []
    train_y.extend(train_y_3node)
    train_y.extend(train_y_4node)

    om = DistDiagnosis_4node(num_classifier=len(labels_to_evaluate), classes_for_each_node=labels_to_evaluate)
    om.train(train_x, train_y)

    test_node_metric = get_test_node_metric(test_node_metric_4node, test_idx)

    root_cause = om.predict(test_x_4node, test_node_metric)

    om_ac_1 = distdiagnosis_ac_at_k(root_cause, test_y_4node, 1)
    om_ac_2 = distdiagnosis_ac_at_k(root_cause, test_y_4node, 2)
    om_ac_3 = distdiagnosis_ac_at_k(root_cause, test_y_4node, 3)
    om_ac_4 = distdiagnosis_ac_at_k(root_cause, test_y_4node, 4)
    om_ac_5 = distdiagnosis_ac_at_k(root_cause, test_y_4node, 5)
    print("overall ac@1", om_ac_1)
    print("overall ac@2", om_ac_2)
    print("overall ac@3", om_ac_3)
    print("overall ac@4", om_ac_4)
    print("overall ac@5", om_ac_5)

    split_test_x = get_split_test_x(test_data_4node, test_idx, len(read_dataset))
    split_test_node_metric = get_split_test_x(test_node_metric_4node, test_idx, len(read_dataset))
    split_test_y = get_split_test_x(test_labels_4node, test_idx, len(read_dataset))
    for i in range(len(split_test_x)):
        root_cause = om.predict(split_test_x[i],split_test_node_metric[i])
        om_ac_1 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 1)
        om_ac_2 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 2)
        om_ac_3 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 3)
        om_ac_4 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 4)
        om_ac_5 = distdiagnosis_ac_at_k(root_cause, split_test_y[i], 5)
        print(f"class {i} ac@1", om_ac_1)
        print(f"class {i} ac@2", om_ac_2)
        print(f"class {i} ac@3", om_ac_3)
        print(f"class {i} ac@4", om_ac_4)
        print(f"class {i} ac@5", om_ac_5)

    return test_idx, train_idx

if __name__ == '__main__':
    print("--------DD_3train_4test-------------")
    test_idx, train_idx = evaluate_distdiagnosis_4_for_4()