## distributed database diagnosis for compound anomalies
A lightweight distributed database diagnostic tool.

## content
The following files are included.

- **data**: It includes utility classes for anomaly data generation and data processing. DistDiagnosis requires training on the deployed database cluster to produce reasonable diagnostic results. 

- **abnormal**: Some predefined anomaly cases.

- **dist_diagnosis**: The source code of DistDiagnosis.

- **monitor**: Class for monitoring the database nodes, e.g. OceanBase OBServer.

- **ob_controller**: A helper class for OB tpc-c benchmark.

- **util**: Some helper classes.


## environment setup

### install python requirements.
**python 3.9** or higher.
```
python -m pip install sklearn tqdm scipy xgboost
```

### install dstat for node monitoring
```
yum install dstat
```

### setup database cluster
Set up [OceanBase 4.3.0](https://www.oceanbase.com/docs/common-oceanbase-database-cn-1000000000695356) for testing, OBProxy is optional.

### set up benchmark suite
Set up benchmark suite, such as [TPC-C](https://www.oceanbase.com/docs/community-tutorials-cn-1000000001390099), [TPC-H](https://www.oceanbase.com/docs/community-tutorials-cn-1000000001390104) etc.

### ensure the configuration is filled out.
configure ```util/config.py```

## transfer DistDiagnosis to other distributed databases

### implement a new monitor
DistDiagnosis is primarily designed for distributed databases composed of homogeneous nodes, with monitoring provided by `monitor`.
`monitor/dstat_monitor.py` uses the `dstat` tool and does not require modifications, while `monitor/node_monitor.py` and `monitor/ob_monitor` need to be adapted for the new database.
