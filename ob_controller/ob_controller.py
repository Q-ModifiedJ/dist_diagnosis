from subprocess import Popen
from util.config import benchmark_path
import os

class OceanBaseController():
    def __init__(self):
        pass

    def ob_stop(self):
        raise NotImplementedError

    def ob_start(self):
        raise NotImplementedError

    def ob_tpccbench_start(self):
        benchmark_run = os.path.join(benchmark_path, 'run')
        cmd = "cd {}; ./runBenchmark.sh prop.oceanbase".format(benchmark_run)
        Popen(cmd, shell=True)
