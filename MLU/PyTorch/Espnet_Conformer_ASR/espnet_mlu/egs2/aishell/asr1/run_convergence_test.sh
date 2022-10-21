workdir=`pwd`
espdir=$workdir/../../..
export PYTHONPATH=$PYTHONPATH:$espdir
export PATH=$PATH:$espdir/tools/SCTK/bin
export NLTK_DATA=$espdir/nltk_data
export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
./run.sh --ngpu 16

echo "LM training performance"
lm_log_path=exp/lm_train_lm_transformer_zh_char/train.log
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
asr_log_path=exp/asr_train_asr_conformer_mlu_raw_zh_char_sp/train.log
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