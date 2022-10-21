set -e

CUR_DIR=$(cd `dirname $0`; pwd)

cndb_submit \
    --db_file $CUR_DIR/config/demo-db.yaml \
    --dev_num 1 \
    --model resnet50 \
    --framework tf \
    --dataset imagenet \
    --dev mlu290 \
    --perf_data '{"throughtput": 123}' \
    --tags 't1,t2'

cndb_submit \
    --model resnet50 \
    --framework tf \
    --dataset imagenet2012 \
    --dev MLU270-X5K \
    --dev_num 2 \
    --batch_size 128 \
    --perf_data '{"mean":56.97288997644099,"min":56.90202125997588,"max":57.03135269041314,"std":0.052576521215684915,"count":5}' \
    --metric_data '{}' \
    --eval_type O1 \
    --train_type O1 \
    --dist_type hvd \
    --hard_file $CUR_DIR/data/demo-hard-info.yaml \
    --soft_file $CUR_DIR/data/demo-soft-info.yaml \
    --save_file /tmp/resnet50-2_128.yaml \
    --tags 't1,t2'

cndb_submit \
    --db_file $CUR_DIR/config/demo-db.yaml \
    --load_file /tmp/resnet50-2_128.yaml \
    --log-level INFO
