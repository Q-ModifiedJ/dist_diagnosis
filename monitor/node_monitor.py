from util import ob_metrics
import csv
import os

class NodeMonitor:

    def __init__(self,
                 node,
                 ob_user,
                 ob_pwd,
                 ob_host,
                 ob_port,
                 ob_db,
                 target_con_id,  # 单个Monitor监控的租户ID
                 svr_ip  # 单个Monitor监控的节点IP
                 ):
        self.node = node
        self.nick = ob_metrics.metrics

        self.metrics_cnt = len(self.nick)

        self.vars = self.nick
        self.type = "d"
        self.width = 5
        self.scale = 1

        self.ob_user = ob_user
        self.ob_pwd = ob_pwd
        self.ob_host = ob_host
        self.ob_port = ob_port
        self.ob_db = ob_db

        self.con_id = target_con_id
        self.svr_ip = svr_ip

        self.prepare()

    def prepare(self):
        args = {}
        args["user"] = self.ob_user
        args["passwd"] = self.ob_pwd
        args["host"] = self.ob_host
        args["port"] = int(self.ob_port)
        args["db"] = self.ob_db
        self.db = pymysql.connect(**args)

        self.data = []  # 二维列表存储的csv，第一行为列名，之后每一行为监控指标
        self.data.append(list(self.vars))

    def extract(self):
        try:
            c = self.db.cursor()
            sql = "select name, value from gv$sysstat where con_id={} and svr_ip=\"{}\";".format(self.con_id,
                                                                                                 self.svr_ip)
            c.execute(sql)
            rows = c.fetchall()

            new_row = {}
            for row in rows:
                name, value = row
                new_row[name] = value

            data_row = []
            for metric in self.vars:
                if metric in new_row:
                    data_row.append(new_row[metric])
                else:
                    data_row.append(-1)
            self.data.append(data_row)

        except Exception as e:
            tmp_row = [-1] * self.metrics_cnt
            self.data.append(tmp_row)

    def to_csv(self, data_dir, file_name):
        file = os.path.join(data_dir, file_name)
        with open(file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.data)
