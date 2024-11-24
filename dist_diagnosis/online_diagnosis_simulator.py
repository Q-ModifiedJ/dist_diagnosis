import time

from dist_diagnosis.dist_diagnosis_3node_demo import *

def simulate_online_diagnosis():
    # 填上诊断的目标类型
    read_dataset = [] #["cpu_2_saturation", "io_2_saturation", "buffer_2", "net_2_saturation", "normal"]
    labels_to_evaluate = []#["cpu", "io", "buffer", "net"]

    random.seed(1314)
    training_labels, training_data, node_metric = prepare_data_for_distdiagnosis(read_dataset, labels_to_evaluate)
    print("Total data: ", len(training_labels))
    train_idx, test_idx = get_split_idx(len(training_labels), 0.80)

    train_x, train_y, test_x, test_y = split_training_test(np.array(training_data), np.array(training_labels),
                                                           train_idx, test_idx)

    test_node_metric = get_test_node_metric(node_metric, test_idx)
    om = DistDiagnosis(num_classifier=len(labels_to_evaluate), classes_for_each_node=labels_to_evaluate)
    om.train(train_x, train_y)

    while True:
        print("Simulating Diagnosis------")
        time.sleep(5)
        root_cause = om.predict(test_x[0:1], test_node_metric)
        print(root_cause)
    return test_idx, train_idx


if __name__ == "__main__":
    simulate_online_diagnosis()