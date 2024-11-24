import time

from util import config
from abnormal.abnormal import IOSaturation, CPUSaturation, HeavyWorkload
from monitor.ob_monitor import OBMonitor
from ob_controller.ob_controller import OceanBaseController

nodes = config.node_4
remote_dstat_path = config.remote_dstat_report_path
data_dir = config.data_4node_dir
def collect_cpu_saturation_data_onenode(number, node):
    assert node in [0,1,2,3]
    labels = ["normal", "normal", "normal", "normal"]
    labels[node] = "cpu"

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, server_num=4)
    abnormal = CPUSaturation(node=nodes[node])

    # 收集一个idle数据
    ob_monitor.prepare(f"{data_dir}/cpu_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    abnormal.trigger()
    ob_monitor.start()

def collect_cpu_saturation_data_twonode(number, node1, node2):
    assert node1 in [0,1,2,3]
    assert node2 in [0,1,2,3]

    labels = ["normal", "normal", "normal", "normal"]
    labels[node1] = "cpu"
    labels[node2] = "cpu"

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, server_num=4)
    abnormal_1 = CPUSaturation(node=nodes[node1])
    abnormal_2 = CPUSaturation(node=nodes[node2])


    # 收集一个idle数据
    ob_monitor.prepare(f"{data_dir}/cpu_2_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    abnormal_1.trigger()
    abnormal_2.trigger()
    ob_monitor.start()

def collect_io_and_cpu_saturation_data_onenode(number, node):
    assert node in [0,1,2,3]
    labels = ["normal", "normal", "normal", "normal"]
    labels[node] = "io"
    extral_label = "node{}.cpu".format(node+1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, server_num=4, extra_label=extral_label)
    io_abnormal = IOSaturation(node=nodes[node])
    cpu_abnormal = CPUSaturation(node=nodes[node])

    # 收集一个idle数据
    ob_monitor.prepare(f"{data_dir}/io_and_cpu_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    io_abnormal.trigger()
    cpu_abnormal.trigger()
    ob_monitor.start()

def collect_cpu_saturation_and_heavyload_data_onenode(number, node):
    assert node in [0,1,2,3]
    labels = ["load", "load", "load", "load"]
    extra_label = "node{}.cpu".format(node+1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels,extra_label=extra_label,server_num=4)
    cpu_abnormal = CPUSaturation(node=nodes[node])
    load_abnormal = HeavyWorkload(0)

    # 收集一个net数据
    ob_monitor.prepare(f"{data_dir}/cpu_saturation_and_load/case{number}")
    ob_controller.ob_tpccbench_start()
    cpu_abnormal.trigger()
    load_abnormal.trigger()
    ob_monitor.start()

def cpu():
    # collect node 1 cpu saturation data
    print("collecting cpu node0")
    for i in range(1, 4):
        collect_cpu_saturation_data_onenode(i, 0)
        time.sleep(20)

    print("collecting cpu node1")
    for i in range(4, 7):
        collect_cpu_saturation_data_onenode(i, 1)
        time.sleep(20)

    print("collecting cpu node2")
    for i in range(7, 9):
        collect_cpu_saturation_data_onenode(i, 2)
        time.sleep(20)

    print("collecting cpu node3")
    for i in range(9, 11):
        collect_cpu_saturation_data_onenode(i, 3)
        time.sleep(20)

def cpu_2():
    # collect 2 node cpu saturation data
    print("collecting cpu node0")
    collect_cpu_saturation_data_twonode(1, 0, 1)
    time.sleep(20)

    print("collecting cpu node0")
    collect_cpu_saturation_data_twonode(2, 0, 1)
    time.sleep(20)

    print("collecting cpu node0")
    collect_cpu_saturation_data_twonode(3, 0, 2)
    time.sleep(20)

    print("collecting cpu node0")
    collect_cpu_saturation_data_twonode(4, 0, 2)
    time.sleep(20)

    print("collecting cpu node0")
    collect_cpu_saturation_data_twonode(5, 0, 3)
    time.sleep(20)

    print("collecting cpu node0")
    collect_cpu_saturation_data_twonode(6, 1, 2)
    time.sleep(20)

    print("collecting cpu node0")
    collect_cpu_saturation_data_twonode(7, 1, 2)
    time.sleep(20)

    print("collecting cpu node0")
    collect_cpu_saturation_data_twonode(8, 2, 3)
    time.sleep(20)

    print("collecting cpu node0")
    collect_cpu_saturation_data_twonode(9, 2, 3)
    time.sleep(20)

    print("collecting cpu node0")
    collect_cpu_saturation_data_twonode(10, 2, 3)
    time.sleep(20)

def cpu_io():
    print("collecting cpu node0")
    for i in range(1, 4):
        collect_io_and_cpu_saturation_data_onenode(i, 0)
        time.sleep(20)

    print("collecting cpu node1")
    for i in range(4, 7):
        collect_io_and_cpu_saturation_data_onenode(i, 1)
        time.sleep(20)

    print("collecting cpu node2")
    for i in range(7, 9):
        collect_io_and_cpu_saturation_data_onenode(i, 2)
        time.sleep(20)

    print("collecting cpu node3")
    for i in range(9, 11):
        collect_io_and_cpu_saturation_data_onenode(i, 3)
        time.sleep(20)

def cpu_and_load():
    print("collecting cpu node0")
    for i in range(1, 4):
        collect_cpu_saturation_and_heavyload_data_onenode(i, 0)
        time.sleep(20)

    print("collecting cpu node1")
    for i in range(4, 7):
        collect_cpu_saturation_and_heavyload_data_onenode(i, 1)
        time.sleep(20)

    print("collecting cpu node2")
    for i in range(7, 9):
        collect_cpu_saturation_and_heavyload_data_onenode(i, 2)
        time.sleep(20)

    print("collecting cpu node3")
    for i in range(9, 11):
        collect_cpu_saturation_and_heavyload_data_onenode(i, 3)
        time.sleep(20)

if __name__ == "__main__":
    cpu()
    cpu_2()
    cpu_io()
    cpu_and_load()