import time
from abnormal.abnormal import IOSaturation, CPUSaturation, NetSaturation, HeavyWorkload
from util import config
from ob_controller.ob_controller import OceanBaseController
from monitor.ob_monitor import OBMonitor

nodes = config.nodes
data_dir = config.data_dir
remote_dstat_path = config.remote_dstat_report_path


def collect_idle_data(number):
    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=["idle", "idle", "idle"])

    # 收集一个idle数据
    ob_monitor.prepare(f"{data_dir}/idle/case{number}")
    # abnormal.trigger()
    ob_monitor.start()


def collect_normal_data(number):
    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=["normal", "normal", "normal"])

    # 正常测试数据
    ob_monitor.prepare(f"{data_dir}/normal/case{number}")
    ob_controller.ob_tpccbench_start()
    ob_monitor.start()


def collect_heavyload_data(number, load_num):
    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=["load", "load", "load"])
    abnormal = HeavyWorkload(load_num)

    # 收集一个heavy load数据
    ob_monitor.prepare(f"{data_dir}/load/case{number}")
    ob_controller.ob_tpccbench_start()
    abnormal.trigger()
    ob_monitor.start()


def collect_toomanyindex_data(number):
    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=["toomanyindex", "toomanyindex", "toomanyindex"])

    # 收集一个too many index数据
    ob_monitor.prepare(f"{data_dir}/toomanyindex/case{number}")
    ob_controller.ob_tpccbench_start()
    ob_monitor.start()


def collect_cpu_saturation_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["normal", "normal", "normal"]
    labels[node] = "cpu"

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels)
    abnormal = CPUSaturation(node=nodes[node])

    # 收集一个idle数据
    ob_monitor.prepare(f"{data_dir}/cpu_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    abnormal.trigger()
    ob_monitor.start()


def collect_cpu_saturation_data_twonode(number, node1, node2):
    assert node1 in [0, 1, 2]
    assert node2 in [0, 1, 2]

    labels = ["normal", "normal", "normal"]
    labels[node1] = "cpu"
    labels[node2] = "cpu"

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels)
    abnormal_1 = CPUSaturation(node=nodes[node1])
    abnormal_2 = CPUSaturation(node=nodes[node2])

    # 收集一个idle数据
    ob_monitor.prepare(f"{data_dir}/cpu_2_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    abnormal_1.trigger()
    abnormal_2.trigger()
    ob_monitor.start()


def collect_io_saturation_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["normal", "normal", "normal"]
    labels[node] = "io"

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels)
    abnormal = IOSaturation(node=nodes[node])

    # 收集一个idle数据
    ob_monitor.prepare(f"{data_dir}/io_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    abnormal.trigger()
    ob_monitor.start()


def collect_io_saturation_data_twonode(number, node1, node2):
    assert node1 in [0, 1, 2]
    assert node2 in [0, 1, 2]

    labels = ["normal", "normal", "normal"]
    labels[node1] = "io"
    labels[node2] = "io"

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels)
    abnormal_1 = IOSaturation(node=nodes[node1])
    abnormal_2 = IOSaturation(node=nodes[node2])

    # 收集一个idle数据
    ob_monitor.prepare(f"{data_dir}/io_2_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    abnormal_1.trigger()
    abnormal_2.trigger()
    ob_monitor.start()


def collect_net_saturation_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["normal", "normal", "normal"]
    labels[node] = "net"

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels)
    abnormal = NetSaturation(node=nodes[node])
    abnormal.recover()
    # 收集一个net数据
    ob_monitor.prepare(f"{data_dir}/net_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    abnormal.trigger()
    ob_monitor.start()
    abnormal.recover()


def collect_net_saturation_data_twonode(number, node1, node2):
    assert node1 in [0, 1, 2]
    assert node2 in [0, 1, 2]

    labels = ["normal", "normal", "normal"]
    labels[node1] = "net"
    labels[node2] = "net"

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels)
    abnormal_1 = NetSaturation(node=nodes[node1])
    abnormal_2 = NetSaturation(node=nodes[node2])
    abnormal_1.recover()
    abnormal_2.recover()
    # 收集一个net数据
    ob_monitor.prepare(f"{data_dir}/net_2_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    abnormal_1.trigger()
    abnormal_2.trigger()
    ob_monitor.start()
    abnormal_1.recover()
    abnormal_2.recover()


def collect_buffer_saturation_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["normal", "normal", "normal"]
    labels[node] = "buffer"

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels)
    abnormal = NetSaturation(node=nodes[node])
    abnormal.recover()
    # 收集一个net数据
    ob_monitor.prepare(f"{data_dir}/buffer/case{number}")
    ob_controller.ob_tpccbench_start()
    ob_monitor.start()


def collect_buffer_saturation_data_twonode(number, node1, node2):
    assert node1 in [0, 1, 2]
    assert node2 in [0, 1, 2]

    labels = ["normal", "normal", "normal"]
    labels[node1] = "buffer"
    labels[node2] = "buffer"

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels)
    abnormal_1 = NetSaturation(node=nodes[node1])
    abnormal_2 = NetSaturation(node=nodes[node2])
    abnormal_1.recover()
    abnormal_2.recover()
    # 收集一个net数据
    ob_monitor.prepare(f"{data_dir}/buffer_2/case{number}")
    ob_controller.ob_tpccbench_start()

    ob_monitor.start()
    abnormal_1.recover()
    abnormal_2.recover()


def cpu():
    # collect node 1 cpu saturation data
    print("collecting cpu node0")
    for i in range(2, 5):
        collect_cpu_saturation_data_onenode(i, 0)
        time.sleep(20)

    print("collecting cpu node1")
    for i in range(5, 8):
        collect_cpu_saturation_data_onenode(i, 1)
        time.sleep(20)

    print("collecting cpu node2")
    for i in range(8, 11):
        collect_cpu_saturation_data_onenode(i, 2)
        time.sleep(20)


def cpu_2():
    # collect 2 node cpu saturation data
    print("collecting cpu node0")
    for i in range(1, 4):
        collect_cpu_saturation_data_twonode(i, 0, 1)
        time.sleep(20)

    print("collecting cpu node1")
    for i in range(4, 8):
        collect_cpu_saturation_data_twonode(i, 0, 2)
        time.sleep(20)

    print("collecting cpu node2")
    for i in range(8, 11):
        collect_cpu_saturation_data_twonode(i, 1, 2)
        time.sleep(20)


def io():
    # collect node 1 io saturation data
    print("collecting io node0")
    for i in range(2, 5):
        collect_io_saturation_data_onenode(i, 0)
        time.sleep(20)

    print("collecting io node1")
    for i in range(5, 8):
        collect_io_saturation_data_onenode(i, 1)
        time.sleep(20)

    print("collecting io node2")
    for i in range(8, 11):
        collect_io_saturation_data_onenode(i, 2)
        time.sleep(20)


def io_2():
    # collect node 2 io saturation data
    print("collecting io node0")
    for i in range(1, 4):
        collect_io_saturation_data_twonode(i, 0, 1)
        time.sleep(20)

    print("collecting io node1")
    for i in range(4, 8):
        collect_io_saturation_data_twonode(i, 0, 2)
        time.sleep(20)

    print("collecting io node2")
    for i in range(8, 11):
        collect_io_saturation_data_twonode(i, 1, 2)
        time.sleep(20)


def net():
    # collect node 1 io saturation data
    print("collecting net node0")
    for i in range(1, 5):
        collect_net_saturation_data_onenode(i, 0)
        time.sleep(20)

    print("collecting net node1")
    for i in range(5, 8):
        collect_net_saturation_data_onenode(i, 1)
        time.sleep(20)

    print("collecting net node2")
    for i in range(8, 11):
        collect_net_saturation_data_onenode(i, 2)
        time.sleep(20)


def net_2():
    # collect node 2 io saturation data
    print("collecting net node0")
    for i in range(1, 5):
        collect_net_saturation_data_twonode(i, 0, 1)
        time.sleep(20)

    print("collecting net node1")
    for i in range(5, 8):
        collect_net_saturation_data_twonode(i, 0, 2)
        time.sleep(20)

    print("collecting net node2")
    for i in range(8, 11):
        collect_net_saturation_data_twonode(i, 1, 2)
        time.sleep(20)


def heavyload():
    pass


def collect_net_and_toomanyindex_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["toomanyindex", "toomanyindex", "toomanyindex"]
    extra_label = "node{}.net".format(node + 1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, extra_label=extra_label)
    abnormal = NetSaturation(node=nodes[node])
    abnormal.recover()
    # 收集一个net数据
    ob_monitor.prepare(f"/root/DistDiagnosis/data/net_saturation_and_toomanyindex/case{number}")
    ob_controller.ob_tpccbench_start()
    abnormal.trigger()
    ob_monitor.start()
    abnormal.recover()


def net_and_toomany():
    # collect node 2 io saturation data
    print("collecting net node0")
    for i in range(1, 5):
        collect_net_and_toomanyindex_data_onenode(i, 0)
        time.sleep(20)

    print("collecting net node1")
    for i in range(5, 8):
        collect_net_and_toomanyindex_data_onenode(i, 1)
        time.sleep(20)

    print("collecting net node2")
    for i in range(8, 11):
        collect_net_and_toomanyindex_data_onenode(i, 2)
        time.sleep(20)


def collect_buffer_and_toomanyindex_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["toomanyindex", "toomanyindex", "toomanyindex"]
    extra_label = "node{}.buffer".format(node + 1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, extra_label=extra_label)

    # 收集一个net数据
    ob_monitor.prepare(f"/root/DistDiagnosis/data/buffer_and_toomanyindex/case{number}")
    ob_controller.ob_tpccbench_start()
    ob_monitor.start()


def collect_io_saturation_and_heavyload_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["load", "load", "load"]
    extra_label = "node{}.io".format(node + 1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, extra_label=extra_label)
    io_abnormal = IOSaturation(node=nodes[node])
    load_abnormal = HeavyWorkload(0)

    # 收集一个net数据
    ob_monitor.prepare(f"/root/DistDiagnosis/data/io_saturation_and_load/case{number}")
    ob_controller.ob_tpccbench_start()
    io_abnormal.trigger()
    load_abnormal.trigger()
    ob_monitor.start()


def io_and_heavy():
    # collect node 2 io saturation data
    print("collecting net node0")
    for i in range(1, 5):
        collect_io_saturation_and_heavyload_data_onenode(i, 0)
        time.sleep(20)

    print("collecting net node1")
    for i in range(5, 8):
        collect_io_saturation_and_heavyload_data_onenode(i, 1)
        time.sleep(20)

    print("collecting net node2")
    for i in range(8, 11):
        collect_io_saturation_and_heavyload_data_onenode(i, 2)
        time.sleep(20)


def collect_cpu_saturation_and_heavyload_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["load", "load", "load"]
    extra_label = "node{}.cpu".format(node + 1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, extra_label=extra_label)
    cpu_abnormal = CPUSaturation(node=nodes[node])
    load_abnormal = HeavyWorkload(0)

    # 收集一个net数据
    ob_monitor.prepare(f"/root/DistDiagnosis/data/cpu_saturation_and_load/case{number}")
    ob_controller.ob_tpccbench_start()
    cpu_abnormal.trigger()
    load_abnormal.trigger()
    ob_monitor.start()


def cpu_and_heavy():
    # collect node 2 io saturation data
    print("collecting net node0")
    for i in range(1, 5):
        collect_cpu_saturation_and_heavyload_data_onenode(i, 0)
        time.sleep(20)

    print("collecting net node1")
    for i in range(5, 8):
        collect_cpu_saturation_and_heavyload_data_onenode(i, 1)
        time.sleep(20)

    print("collecting net node2")
    for i in range(8, 11):
        collect_cpu_saturation_and_heavyload_data_onenode(i, 2)
        time.sleep(20)


def collect_io_and_cpu_saturation_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["normal", "normal", "normal"]
    labels[node] = "io"
    extra_label = "node{}.cpu".format(node + 1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, extra_label=extra_label)
    io_abnormal = IOSaturation(node=nodes[node])
    cpu_abnormal = CPUSaturation(node=nodes[node])

    # 收集一个idle数据
    ob_monitor.prepare(f"/root/DistDiagnosis/data/io_and_cpu_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    io_abnormal.trigger()
    cpu_abnormal.trigger()
    ob_monitor.start()


def collect_cpu_and_net_saturation_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["normal", "normal", "normal"]
    labels[node] = "net"
    extra_label = "node{}.cpu".format(node + 1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, extra_label=extra_label)
    net_abnormal = NetSaturation(node=nodes[node])
    cpu_abnormal = CPUSaturation(node=nodes[node])
    net_abnormal.recover()

    # 收集一个idle数据
    ob_monitor.prepare(f"/root/DistDiagnosis/data/cpu_and_net_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    net_abnormal.trigger()
    cpu_abnormal.trigger()
    ob_monitor.start()
    net_abnormal.recover()


def collect_io_and_net_saturation_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["normal", "normal", "normal"]
    labels[node] = "io"
    extra_label = "node{}.net".format(node + 1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, extra_label=extra_label)
    net_abnormal = NetSaturation(node=nodes[node])
    io_abnormal = IOSaturation(node=nodes[node])
    net_abnormal.recover()

    # 收集一个idle数据
    ob_monitor.prepare(f"/root/DistDiagnosis/data/io_and_net_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    net_abnormal.trigger()
    io_abnormal.trigger()
    ob_monitor.start()
    net_abnormal.recover()


def collect_cpu_saturation_and_heavyload_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["load", "load", "load"]
    extra_label = "node{}.cpu".format(node + 1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, extra_label=extra_label)
    cpu_abnormal = CPUSaturation(node=nodes[node])
    load_abnormal = HeavyWorkload(0)

    # 收集一个net数据
    ob_monitor.prepare(f"/root/DistDiagnosis/data/cpu_saturation_and_load/case{number}")
    ob_controller.ob_tpccbench_start()
    cpu_abnormal.trigger()
    load_abnormal.trigger()
    ob_monitor.start()


def collect_buffer_and_io_saturation_data_onenode(number, node):
    assert node in [0, 1, 2]
    labels = ["normal", "normal", "normal"]
    labels[node] = "io"
    extra_label = "node{}.buffer".format(node + 1)

    ob_controller = OceanBaseController()
    ob_monitor = OBMonitor(30, 5, labels=labels, extra_label=extra_label)
    io_abnormal = IOSaturation(node=nodes[node])

    # 收集一个idle数据
    ob_monitor.prepare(f"/root/DistDiagnosis/data/buffer_and_io_saturation/case{number}")
    ob_controller.ob_tpccbench_start()
    io_abnormal.trigger()
    ob_monitor.start()


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
    for i in range(7, 11):
        collect_io_and_cpu_saturation_data_onenode(i, 2)
        time.sleep(20)


def cpu_net():
    print("collecting cpu node0")
    for i in range(1, 4):
        collect_cpu_and_net_saturation_data_onenode(i, 0)
        time.sleep(20)

    print("collecting cpu node1")
    for i in range(4, 7):
        collect_cpu_and_net_saturation_data_onenode(i, 1)
        time.sleep(20)

    print("collecting cpu node2")
    for i in range(7, 11):
        collect_cpu_and_net_saturation_data_onenode(i, 2)
        time.sleep(20)


def io_net():
    print("collecting cpu node0")
    for i in range(1, 4):
        collect_io_and_net_saturation_data_onenode(i, 0)
        time.sleep(20)

    print("collecting cpu node1")
    for i in range(4, 7):
        collect_io_and_net_saturation_data_onenode(i, 1)
        time.sleep(20)

    print("collecting cpu node2")
    for i in range(7, 11):
        collect_io_and_net_saturation_data_onenode(i, 2)
        time.sleep(20)


if __name__ == '__main__':
    cpu() # 收集CPU异常下的10个Case数据， 其他Case同理

# obd test tpcc myoceanbase --test-only --tenant=tpcc --warehouses 10 --run-mins 1 --password=AAaa12345678+= -O0

"""
正常情况、三节点、单节点请求、10 warehouses，60terminal，40loadworkers
02:28:07,427 [Thread-59] INFO   jTPCC : Term-00, 
02:28:07,427 [Thread-59] INFO   jTPCC : Term-00, 
02:28:07,427 [Thread-59] INFO   jTPCC : Term-00, Measured tpmC (NewOrders) = 62442.54
02:28:07,427 [Thread-59] INFO   jTPCC : Term-00, Measured tpmTOTAL = 138989.58
02:28:07,427 [Thread-59] INFO   jTPCC : Term-00, Session Start     = 2024-07-25 02:27:07
02:28:07,428 [Thread-59] INFO   jTPCC : Term-00, Session End       = 2024-07-25 02:28:07
02:28:07,428 [Thread-59] INFO   jTPCC : Term-00, Transaction Count = 139299
"""
