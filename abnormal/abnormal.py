import os
import time
from subprocess import Popen

class CPUSaturation():
    def __init__(self, node):
        self.node = node

    def trigger(self):
        ip = self.node
        #Popen(cmd = , shell=True)
        os.system(f"ssh {ip} stress-ng --cpu 30 -t 180 &")


class IOSaturation():
    def __init__(self, node):
        self.node = node

    def trigger(self):
        ip = self.node
        #Popen(cmd = , shell=True)
        os.system(f"ssh {ip} stress-ng --io 320 -t 180 &")


class NetSaturation():
    def __init__(self, node):
        self.node = node
        self.exec = os.system
    # 采用tc阻塞住主机的2881端口、
    # tc阻塞集群间远程连接的2882端口
    # 2881 客户端连接
    # 2882
    def trigger(self):
        ip = self.node
        cmd = f"ssh {ip} 'tc qdisc add dev eth0 root handle 1: htb default 1'"
        self.exec(cmd)

        cmd = f"ssh {ip} 'tc class add dev eth0 parent 1: classid 1:1 htb rate 1mbps'"
        self.exec(cmd)

        cmd = f"ssh {ip} 'tc class add dev eth0 parent 1:1 classid 1:2 htb rate 100kbit ceil 120kbit prio 1'"
        self.exec(cmd)

        cmd = f"ssh {ip} 'tc class add dev eth0 parent 1:1 classid 1:3 htb rate 100kbit ceil 120kbit prio 1'"
        self.exec(cmd)

        cmd = f"ssh {ip} 'tc filter add dev eth0 parent 1:0 prio 1 protocol ip handle 2 fw flowid 1:2'"
        self.exec(cmd)

        cmd = f"ssh {ip} 'tc filter add dev eth0 parent 1:0 prio 1 protocol ip handle 3 fw flowid 1:3'"
        self.exec(cmd)

        cmd = f"ssh {ip} 'iptables -A OUTPUT -t mangle -p tcp --sport 2881 -j MARK --set-mark 2'"
        self.exec(cmd)

        cmd = f"ssh {ip} 'iptables -A OUTPUT -t mangle -p tcp --sport 2882 -j MARK --set-mark 3'"
        self.exec(cmd)


    def recover(self):
        ip = self.node
        cmd = f"ssh {ip} 'iptables -t mangle -F'"
        self.exec(cmd)

        cmd = f"ssh {ip} 'tc qdisc del dev eth0 root'"
        self.exec(cmd)

# 一些异常需要手动触发
class HeavyWorkload():

    def __init__(self, load_num):
        self.load_num = load_num

    def trigger(self):

        pass


class Dump():
    def __init__(self):
        pass

    def trigger(self):
        # cmd = "JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.412.b08-1.el7_9.x86_64/jre; /root/DistDiagnosis/ob-loader-dumper-4.3.0-RELEASE/bin/obdumper -h 10.206.16.16 -P2881 -u root -t tpcc -p AAaa12345678+= -D test --csv --all --skip-check-dir -f ~/dump-1"
        # p = Popen("bash /root/start_dump.sh", shell=True)
        # print(cmd)
        # os.system("bash /root/start_dump.sh")
        # return p
        pass

class SmallBufferSize():
    def __init__(self):
        pass

    def trigger(self):
        pass

class TooManyIndex():
    def __init__(self):
        pass

    def trigger(self):
        pass

if __name__ == "__main__":
    x = Dump()
    x.trigger()
