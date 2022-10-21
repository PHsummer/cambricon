set -e

usage() {
  echo "Usage:"
  echo "    ./docker_build.sh -t <type>"
  exit -1
}

CUR_DIR=$(cd `dirname $0`; pwd)
DEL_TYPE=""

while getopts 't:' OPT; do
    case $OPT in
        t) DEL_TYPE="$OPTARG";;
        ?) usage
    esac
done

docker_args=""
docker_name_base="yellow.hub.cambricon.com/distribute_platform"
if [ "$DEL_TYPE" = "dev" ]; then
    docker_file="${CUR_DIR}/dockerfile.dev"
    docker_name="${docker_name_base}/cndb-dev:ubuntu1804"
elif [ "$DEL_TYPE" = "release" ]; then
    docker_file="${CUR_DIR}/dockerfile.release"
    version=$(PYTHONPATH=$CUR_DIR/.. python -c 'import cndb; print(cndb.__version__)')
    docker_name="${docker_name_base}/cndb-release:${version}"
else
    usage
fi

docker_args="-f ${docker_file} ${docker_args}"
cmd="docker build --network=host \
    -t ${docker_name} \
    ${docker_args} ."
echo "Build docker: ${cmd}"
${cmd} 2>&1 | tee result.log

if [ $? != 0 ];then
    echo "DOCKER BUILD FAILD: ${cmd}" >> result.log
    exit -1
else
    echo "DOCKER BUILD SUCCEED: ${cmd}" >> result.log
fi
