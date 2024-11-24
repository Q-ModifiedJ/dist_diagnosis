
metrics = (
            "rpc packet in",
            "rpc packet in bytes",
            "rpc packet out",
            "rpc packet out bytes",
            "rpc deliver fail",
            "rpc net delay",
            "rpc net frame delay",
            "mysql packet in",
            "mysql packet in bytes",
            "mysql packet out",
            "mysql packet out bytes",
            "mysql deliver fail",
            "rpc compress original packet cnt",
            "rpc compress compressed packet cnt",
            "rpc compress original size",
            "rpc compress compressed size",
            "rpc stream compress original packet cnt",
            "rpc stream compress compressed packet cnt",
            "rpc stream compress original size",
            "rpc stream compress compressed size",
            "request enqueue count",
            "request dequeue count",
            "request queue time",
            "trans commit log sync time",
            "trans commit log sync count",
            "trans commit log submit count",
            "trans system trans count",
            "trans user trans count",
            "trans start count",
            "trans total used time",
            "trans commit count",
            "trans commit time",
            "trans rollback count",
            "trans rollback time",
            "trans timeout count",
            "trans local trans count",
            "trans distribute trans count",
            "trans without participant count",
            "redo log replay count",
            "redo log replay time",
            "prepare log replay count",
            "prepare log replay time",
            "commit log replay count",
            "commit log replay time",
            "abort log replay count",
            "abort log replay time",
            "clear log replay count",
            "clear log replay time",
            "gts request total count",
            "gts acquire total time",
            "gts acquire total wait count",
            "gts wait elapse total time",
            "gts wait elapse total count",
            "gts wait elapse total wait count",
            "gts rpc count",
            "gts try acquire total count",
            "gts try wait elapse total count",
            "trans early lock release enable count",
            "trans early lock release unable count",
            "read elr row count",
            "local trans total used time",
            "distributed trans total used time",
            "tx data hit mini cache count",
            "tx data hit kv cache count",
            "tx data read tx ctx count",
            "tx data read tx data memtable count",
            "tx data read tx data sstable count",
            "sql select count",
            "sql select time",
            "sql insert count",
            "sql insert time",
            "sql replace count",
            "sql replace time",
            "sql update count",
            "sql update time",
            "sql delete count",
            "sql delete time",
            "sql other count",
            "sql other time",
            "ps prepare count",
            "ps prepare time",
            "ps execute count",
            "ps close count",
            "ps close time",
            "opened cursors current",
            "opened cursors cumulative",
            "sql local count",
            "sql remote count",
            "sql distributed count",
            "active sessions",
            "single query count",
            "multiple query count",
            "multiple query with one stmt count",
            "sql inner select count",
            "sql inner select time",
            "sql inner insert count",
            "sql inner insert time",
            "sql inner replace count",
            "sql inner replace time",
            "sql inner update count",
            "sql inner update time",
            "sql inner delete count",
            "sql inner delete time",
            "sql inner other count",
            "sql inner other time",
            "user logons cumulative",
            "user logouts cumulative",
            "user logons failed cumulative",
            "user logons time cumulative",
            "sql local execute time",
            "sql remote execute time",
            "sql distributed execute time",
            "row cache hit",
            "row cache miss",
            "bloom filter cache hit",
            "bloom filter cache miss",
            "bloom filter filts",
            "bloom filter passes",
            "block cache hit",
            "block cache miss",
            "location cache hit",
            "location cache miss",
            "location cache wait",
            "location cache get hit from proxy virtual table",
            "location cache get miss from proxy virtual table",
            "location cache nonblock renew",
            "location cache nonblock renew ignored",
            "location nonblock get hit",
            "location nonblock get miss",
            "location cache rpc renew count",
            "location cache renew",
            "location cache renew ignored",
            "kvcache sync wash time",
            "kvcache sync wash count",
            "location cache rpc renew fail count",
            "location cache sql renew count",
            "location cache ignore rpc renew count",
            "fuse row cache hit",
            "fuse row cache miss",
            "schema cache hit",
            "schema cache miss",
            "tablet ls cache hit",
            "tablet ls cache miss",
            "index clog cache hit",
            "index clog cache miss",
            "user tab col stat cache hit",
            "user tab col stat cache miss",
            "user table stat cache hit",
            "user table stat cache miss",
            "opt table stat cache hit",
            "opt table stat cache miss",
            "opt column stat cache hit",
            "opt column stat cache miss",
            "tmp page cache hit",
            "tmp page cache miss",
            "tmp block cache hit",
            "tmp block cache miss",
            "secondary meta cache hit",
            "secondary meta cache miss",
            "opt ds stat cache hit",
            "opt ds stat cache miss",
            "storage meta cache hit",
            "storage meta cache miss",
            "tablet cache hit",
            "tablet cache miss",
            "schema history cache hit",
            "schema history cache miss",
            "io read count",
            "io read delay",
            "io read bytes",
            "io write count",
            "io write delay",
            "io write bytes",
            "memstore scan count",
            "memstore scan succ count",
            "memstore scan fail count",
            "memstore get count",
            "memstore get succ count",
            "memstore get fail count",
            "memstore apply count",
            "memstore apply succ count",
            "memstore apply fail count",
            "memstore get time",
            "memstore scan time",
            "memstore apply time",
            "memstore read lock succ count",
            "memstore read lock fail count",
            "memstore write lock succ count",
            "memstore write lock fail count",
            "memstore wait write lock time",
            "memstore wait read lock time",
            "io read micro index count",
            "io read micro index bytes",
            "io prefetch micro block count",
            "io prefetch micro block bytes",
            "io read uncompress micro block count",
            "io read uncompress micro block bytes",
            "storage read row count",
            "storage delete row count",
            "storage insert row count",
            "storage update row count",
            "memstore row purge count",
            "memstore row compaction count",
            "io read queue delay",
            "io write queue delay",
            "io read callback queuing delay",
            "io read callback process delay",
            "bandwidth in throttle size",
            "bandwidth out throttle size",
            "memstore read row count",
            "ssstore read row count",
            "memstore write lock wakenup count in lock_wait_mgr",
            "exist row effect read",
            "exist row empty read",
            "get row effect read",
            "get row empty read",
            "scan row effect read",
            "scan row empty read",
            "bandwidth in sleep us",
            "bandwidth out sleep us",
            "memstore write lock wait timeout count",
            "accessed data micro block count",
            "data micro block cache hit",
            "accessed index micro block count",
            "index micro block cache hit",
            "blockscaned data micro block count",
            "blockscaned row count",
            "storage filtered row count",
            "backup io read count",
            "backup io read bytes",
            "backup io write count",
            "backup io write bytes",
            "backup delete count",
            "backup io read delay",
            "backup io write delay",
            "cos io read delay",
            "cos io write delay",
            "cos io list delay",
            "backup delete delay",
            "backup io list count",
            "backup io read failed count",
            "backup io write failed count",
            "backup io delete failed count",
            "backup io list failed count",
            "backup io tagging count",
            "backup io tagging failed count",
            "cos io write bytes",
            "cos io write count",
            "cos delete count",
            "cos delete delay",
            "cos list io limit count",
            "cos io list count",
            "cos io read count",
            "cos io read bytes",
            "refresh schema count",
            "refresh schema time",
            "inner sql connection execute count",
            "inner sql connection execute time",
            "log stream table operator get count",
            "log stream table operator get time",
            "palf write io count to disk",
            "palf write size to disk",
            "palf write total time to disk",
            "palf read count from cache",
            "palf read size from cache",
            "palf read total time from cache",
            "palf read io count from disk",
            "palf read size from disk",
            "palf read total time from disk",
            "palf handle rpc request count",
            "archive read log size",
            "archive write log size",
            "restore read log size",
            "restore write log size",
            "clog trans log total size",
            "external log service fetch log size",
            "external log service fetch log count",
            "external log service fetch rpc count",
            "success rpc process",
            "failed rpc process",
            "balancer succ execute count",
            "balancer failed execute count",
            "single retrieve execute count",
            "single retrieve execute time",
            "multi retrieve execute count",
            "multi retrieve execute time",
            "multi retrieve rows",
            "single insert execute count",
            "single insert execute time",
            "multi insert execute count",
            "multi insert execute time",
            "multi insert rows",
            "single delete execute count",
            "single delete execute time",
            "multi delete execute count",
            "multi delete execute time",
            "multi delete rows",
            "single update execute count",
            "single update execute time",
            "multi update execute count",
            "multi update execute time",
            "multi update rows",
            "single insert_or_update execute count",
            "single insert_or_update execute time",
            "multi insert_or_update execute count",
            "multi insert_or_update execute time",
            "multi insert_or_update rows",
            "single replace execute count",
            "single replace execute time",
            "multi replace execute count",
            "multi replace execute time",
            "multi replace rows",
            "batch retrieve execute count",
            "batch retrieve execute time",
            "batch retrieve rows",
            "batch hybrid execute count",
            "batch hybrid execute time",
            "batch hybrid insert_or_update rows",
            "table api login count",
            "table api OB_TRANSACTION_SET_VIOLATION count",
            "query count",
            "query time",
            "query row count",
            "query_and_mutate count",
            "query_and_mutate time",
            "query_and_mutate row count",
            "hbase scan count",
            "hbase scan time",
            "hbase scan row count",
            "hbase put count",
            "hbase put time",
            "hbase put row count",
            "hbase delete count",
            "hbase delete time",
            "hbase delete row count",
            "hbase append count",
            "hbase append time",
            "hbase append row count",
            "hbase increment count",
            "hbase increment time",
            "hbase increment row count",
            "hbase check_put count",
            "hbase check_put time",
            "hbase check_put row count",
            "hbase check_delete count",
            "hbase check_delete time",
            "hbase check_delete row count",
            "hbase hybrid count",
            "hbase hybrid time",
            "hbase hybrid row count",
            "single increment execute count",
            "single increment execute time",
            "multi increment execute count",
            "multi increment execute time",
            "multi increment rows",
            "single append execute count",
            "single append execute time",
            "multi append execute count",
            "multi append execute time",
            "multi append rows",
            "DB time",
            "DB CPU",
            "DB inner sql time",
            "DB inner sql CPU",
            "background elapsed time",
            "background cpu time",
            "(backup/restore) cpu time",
            "Tablespace encryption elapsed time",
            "Tablespace encryption cpu time",
            "wr snapshot task elapse time",
            "wr snapshot task cpu time",
            "wr purge task elapse time",
            "wr purge task cpu time",
            "wr schedular elapse time",
            "wr schedular cpu time",
            "wr user submit snapshot elapse time",
            "wr user submit snapshot cpu time",
            "wr collected active session history row count",
            "ash schedular elapse time",
            "standby fetch log bytes",
            "standby fetch log bandwidth limit",
            "location cache size",
            "tablet ls cache size",
            "table stat cache size",
            "table col stat cache size",
            "index block cache size",
            "sys block cache size",
            "user block cache size",
            "user row cache size",
            "bloom filter cache size",
            "active memstore used",
            "total memstore used",
            "major freeze trigger",
            "memstore limit",
            "min memory size",
            "max memory size",
            "memory usage",
            "min cpus",
            "max cpus",
            "cpu usage",
            "disk usage",
            "observer memory used size",
            "observer memory free size",
            "is mini mode",
            "observer memory hold size",
            "worker time",
            "cpu time",
            "effective observer memory limit",
            "effective system memory",
            "effective hidden sys memory",
            "max session num",
            "oblogger log bytes",
            "election log bytes",
            "rootservice log bytes",
            "oblogger total log count",
            "election total log count",
            "rootservice total log count",
            "async error log dropped count",
            "async warn log dropped count",
            "async info log dropped count",
            "async trace log dropped count",
            "async debug log dropped count",
            "async log flush speed",
            "async generic log write count",
            "async user request log write count",
            "async data maintain log write count",
            "async root service log write count",
            "async schema log write count",
            "async force allow log write count",
            "async generic log dropped count",
            "async user request log dropped count",
            "async data maintain log dropped count",
            "async root service log dropped count",
            "async schema log dropped count",
            "async force allow log dropped count",
            "observer partition table updater user table queue size",
            "observer partition table updater system table queue size",
            "observer partition table updater core table queue size",
            "rootservice start time",
        )