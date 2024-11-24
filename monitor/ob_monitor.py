import time
from node_monitor import NodeMonitor
from util.config import nodes, ob_user, ob_pwd, ob_host, ob_port, ob_db, target_con_id
import os

class OBMonitor():
    def __init__(self,
                 total_extracts,
                 interval,
                 labels,  # 异常数据标签
                 server_num,  # OBServer个数
                 extra_label=None,
                 ):
        self.total_extracts = total_extracts
        self.interval = interval
        self.labels = labels
        self.extra_label = extra_label
        self.node_monitors = []
        self.server_num = server_num

    def prepare(self, data_dir):
        print("Hello World")

        for node_num in range(1, self.server_num + 1):
            new_node_monitor = NodeMonitor(node_num,
                                           ob_user=ob_user,
                                           ob_pwd=ob_pwd,
                                           ob_host=ob_host,
                                           ob_port=ob_port,
                                           ob_db=ob_db,
                                           target_con_id=target_con_id,
                                           svr_ip=nodes[node_num-1]
                                           )
            self.node_monitors.append(new_node_monitor)

        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.dstat_m = dstat_monitor(nodes, self.interval, self.total_extracts - 1, self.data_dir)

    def write_label(self):
        label_file = os.path.join(self.data_dir, "label.txt")
        label_str = ""
        for i in range(len(self.node_monitors)):
            label_str += "node" + str(i+1) + ".{}\n"

        with open(label_file, "w") as f:
            if self.extra_label == None:
                f.write(label_str.format(*self.labels))
            else:
                f.write(str(label_str+"{}").format(*self.labels, self.extra_label))

    def start(self):
        self.dstat_m.start_monitor()

        total_extracts = self.total_extracts
        current = 0
        interval = self.interval
        start = time.time() - 5

        while current < total_extracts:
            now = time.time()
            if now - start > interval:
                for nm in self.node_monitors:
                    nm.extract()
                current += 1
                start = now
                print("--extracting--")

        time.sleep(5)
        for i in range(len(self.node_monitors)):
            self.node_monitors[i].to_csv(self.data_dir, 'ob_report{}.csv'.format(i+1))

        # time.sleep(15)
        self.dstat_m.get_report()
        self.write_label()
