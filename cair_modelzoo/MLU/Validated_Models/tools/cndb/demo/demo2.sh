set -e

CUR_DIR=$(cd `dirname $0`; pwd)

pushd $CUR_DIR
python demo.py \
    --config ./config/demo-db.yaml \
    --model ./data/demo-model.yaml \
    --platform ./data/demo-platform.yaml \
    --result ./data/demo-gpu-result.yaml
popd
