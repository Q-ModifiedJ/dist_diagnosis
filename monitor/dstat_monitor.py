import os
from util import config

remote_dstat_path = config.remote_dstat_report_path


class dstat_monitor:
    def __init__(self, nodes, interval, count, local_report_dir):
        self.nodes = nodes
        self.interval = interval
        self.count = count
        self.local_report_dir = local_report_dir

    def start_monitor(self):

        # 删除上一次的dstat报告文件
        for node in self.nodes:
            os.system("ssh {} rm -rf {}".format(node, remote_dstat_path))

        # 执行监控
        for node in self.nodes:
            cmd = "ssh {} dstat -clmdnp -o {} {} {} &".format(node, remote_dstat_path, self.interval, self.count)
            print(cmd)  # just for debug
            # Popen(cmd, shell=True)
            os.system(cmd)

    def get_report(self):
        i = 1
        for node in self.nodes:
            local_report = "dstat_report{}.csv".format(i)
            file = os.path.join(self.local_report_dir, local_report)
            scp_cmd = "scp root@{}:{} {}".format(node, remote_dstat_path, file)
            os.system(scp_cmd)
            i += 1
