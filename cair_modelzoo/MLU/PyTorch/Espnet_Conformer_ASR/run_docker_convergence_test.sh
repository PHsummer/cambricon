
ESPNET_HOME=`pwd`
AISHELL_DIR="/data/datasets/aishell"
OUTPUT_DIR="$ESPNET_HOME/espnet_output"
CONF_DIR="$ESPNET_HOME/espnet_config"
CACHE_DIR="$ESPNET_HOME/espnet_cache"
CONT_NAME="conformer_pytorch_1.9.0_train"

if ! [[ -d "$OUTPUT_DIR" ]]; then
    echo "Output dir  does not exist, creating $OUTPUT_DIR"
    mkdir -p $OUTPUT_DIR
fi
if ! [[ -d "$CACHE_DIR" ]]; then
    echo "Cache dir does not exist, creating $CACHE_DIR"
    mkdir -p $CACHE_DIR
fi
if ! [[ -d "$CACHE_DIR/dump" ]]; then
    mkdir -p $CACHE_DIR/dump
fi
if ! [[ -d "$CACHE_DIR/data" ]]; then
    mkdir -p $CACHE_DIR/data
fi
if ! [[ -d "$CACHE_DIR/downloads" ]]; then
    mkdir -p $CACHE_DIR/downloads
fi

docker build -t conformer_pytorch_1.9.0_docker:final -f Dockerfile .
docker run -it --rm --name ${CONT_NAME} --device /dev/cambricon_ctl \
        --shm-size '256gb' --ipc=host \
        -v /usr/bin/cnmon:/usr/bin/cnmon \
        -v $AISHELL_DIR:/test_espnet/espnet_mlu/egs2/aishell/asr1/downloads \
        -v $OUTPUT_DIR:/test_espnet/espnet_mlu/egs2/aishell/asr1/exp \
        -v $CONF_DIR:/test_espnet/espnet_mlu/egs2/aishell/asr1/conf/cambricon \
        -v $CACHE_DIR/dump:/test_espnet/espnet_mlu/egs2/aishell/asr1/dump \
        -v $CACHE_DIR/data:/test_espnet/espnet_mlu/egs2/aishell/asr1/data \
        --privileged=true -d conformer_pytorch_1.9.0_docker:final /bin/bash
docker exec -ti --use-nas-user ${CONT_NAME} /bin/bash -c \
        "source /torch/venv3/pytorch/bin/activate && \
        export PYTHONPATH=\$PYTHONPATH:/test_espnet/espnet_mlu && \
        export PATH=\$PATH:/test_espnet/espnet_mlu/tools/SCTK/bin && \
        export NLTK_DATA=/test_espnet/espnet_mlu/nltk_data && \
        export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 && \
        ./run.sh --ngpu 16"
docker container rm -f ${CONT_NAME} || true

echo "LM training performance"
lm_log_path=$OUTPUT_DIR/lm_train_lm_transformer_zh_char/train.log
batch_bins=$(grep "batch_bins" $lm_log_path | head -n 1 | grep -o "batch_bins=[.0-9]*" | grep -o "[.0-9]*")
train_time=$(grep "10epoch results" $lm_log_path | grep -o "train_time=[.0-9]*" | grep -o "[.0-9]*")
e2e_time=$(grep -o "elapsed time [.0-9]*" $lm_log_path | grep -o "[.0-9]*")
echo "batch_bins: $batch_bins bins/batch"
throughput=$(echo "$batch_bins/$train_time" | bc)
echo "throughput: $throughput bins/s"
h=$(echo "$e2e_time/3600" | bc)
m=$(echo "$e2e_time%3600/60" | bc)
s=$(echo "$e2e_time%60" | bc)
echo "e2e_time: $h h $m m $s s ($e2e_time s)"

echo "ASR training performance"
asr_log_path=$OUTPUT_DIR/asr_train_asr_conformer_mlu_raw_zh_char_sp/train.log
batch_bins=$(grep "batch_bins" $asr_log_path | head -n 1 | grep -o "batch_bins=[.0-9]*" | grep -o "[.0-9]*")
train_time=$(grep "10epoch results" $asr_log_path | grep -o "train_time=[.0-9]*" | grep -o "[.0-9]*")
e2e_time=$(grep -o "elapsed time [.0-9]*" $asr_log_path | grep -o "[.0-9]*")
echo "batch_bins: $batch_bins bins/batch"
throughput=$(echo "$batch_bins/$train_time" | bc)
echo "throughput: $throughput bins/s"
h=$(echo "$e2e_time/3600" | bc)
m=$(echo "$e2e_time%3600/60" | bc)
s=$(echo "$e2e_time%60" | bc)
echo "e2e_time: $h h $m m $s s ($e2e_time s)"