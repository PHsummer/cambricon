# cndb

本项目负责将训练的结果上传至 PostgresDB 中，用于展示。

CNDB 的数据库设计可以参考 [cndb-def](./docs/design/cndb-def.yaml)。

CNDB 提供两种使用模式：

1. 通过 cndb_submit 提交数据

    安装 cndb 后，可运行 `cndb_submit ...` 将数据提交到对应数据库中，一次仅能提交一条数据记录。
    可通过运行 `cndb_submit --help` 查看使用方式。
    相关 demo 可以参考 [demo1](./demo/demo1.sh)。

2. 通过 `import cndb`，自定义数据格式进行提交

    cndb 可以作为一个 python 模块使用，通过导入 DBHandler 对数据库进行操作。

    ```python
    from cndb.db import DBHandler
    ```

    相关 demo 可以参考 [demo2](./demo/demo2.sh)

## 安装

运行如下命令完成安装

```bash
python setup.py install
```

## Docker

构建 Release Docker

```bash
bash ./docker/build.sh -t release
```

示例（假设 [demo-db.yaml](./demo/config/demo-db.yaml) 中定义的数据库可用）

```bash
docker run -it \
    -v `pwd`/demo/config:/config \
    yellow.hub.cambricon.com/distribute_platform/cndb:release-0.1.0 \
    cndb_submit \
    --db_file /config/demo-db.yaml \
    --dev_num 1 \
    --model resnet50 \
    --framework tf \
    --dataset imagenet \
    --dev mlu290 \
    --perf_data '{"throughtput": 123}'
```
