usage() {
    echo "Usage:"
    echo "validate_all_models.sh -m modes [-c classes]"
    echo "Description:"
    echo "modes: One or several parameters to run test_*.sh."
    echo "classes: The class of models"
    echo "Example:"
    echo "Validate all models with 4 modes: ./validate_all_models.sh -m 'fp32-mlu fp32-mlu-ddp amp-mlu amp-mlu-ddp'"
    echo "Validate classification and detection models with 1 mode: ./validate_all_models.sh -m fp32-mlu -c 'Classification Detection'"
    exit -1
}

modes=""

while getopts 'm:c:' OPT; do
    case $OPT in
        m) modes="$OPTARG";;
        c) class_list="$OPTARG";;
        h) usage;;
        ?) usage;;
    esac
done
if [[ -z $modes ]]; then
    echo "Validate mode is required."
    usage
    exit -1
fi

model_zoo_home=`pwd`/..
if [[ -z $class_list ]]; then
    class_list=`ls $model_zoo_home/PyTorch`
fi
echo "Validate modes: $modes"
echo "Validate model classes: $class_list"

for class in $class_list 
do
    class_dir=$model_zoo_home/PyTorch/$class
    model_count=`find $class_dir| grep '_Performance.sh' | wc -l`
    echo "Validating $class models, $model_count models to go."
    cd $class_dir
    test_script=`ls | grep -o "test_.*sh"`
    for ((model=0;model<model_count;model++))
    do
        for mode in $modes
        do 
            cmd="./$test_script $model $mode"
            echo $cmd
            $cmd
        done
    done
done
cd $model_zoo_home/tools
