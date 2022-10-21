# Upload result to Superset
model_zoo_home=`pwd`/..
RESULT_DIR=$model_zoo_home/results
# Convert benchmark_log to cndb format
cd $model_zoo_home
for file in `ls $RESULT_DIR`
do
    split=(${file//_/ }) 
    device=${split[1]}
    pcie=${split[2]}
    ctr=${split[3]}
    pt=${split[4]}
    echo "$device, $pcie, $ctr, $pt"
    if [[ $device = "mlu" ]];then
        if [[ $ctr = "ctr2.4.0" ]] && [[ $pt = "pt19" ]];then
            BASE_IMAGE_NAME="yellow.hub.cambricon.com/pytorch/pytorch:v1.3.0-torch1.9-ubuntu18.04"
        elif [[ $ctr = "ctr2.4.2" ]] && [[ $pt = "pt19" ]];then
            BASE_IMAGE_NAME="yellow.hub.cambricon.com/pytorch/pytorch:v1.4.0-torch1.9-ubuntu18.04"
        elif [[ $ctr = "ctr2.4.2" ]] && [[ $pt = "pt16" ]];then
            BASE_IMAGE_NAME="yellow.hub.cambricon.com/pytorch/pytorch:v1.4.0-torch1.6-ubuntu18.04"
        else
            echo "Unable to find base image information"
        fi
    elif [[ $device == "gpu" ]];then
        if [[ $ctr = "cuda11.0.194" ]] && [[ $pt = "pt16" ]];then
            BASE_IMAGE_NAME="nvcr.io/nvidia/pytorch:20.07-py3"
        elif [[ $ctr = "cuda11.3.1" ]] && [[ $pt = "pt19" ]];then
            BASE_IMAGE_NAME="nvcr.io/nvidia/pytorch:21.06-py3"
        else
            echo "Unable to find base image information"
        fi
    fi
    python ${model_zoo_home}/tools/dump.py \
        -i ${RESULT_DIR}/${file} \
        -o ${RESULT_DIR} \
        --framework $BASE_IMAGE_NAME  \
        --code_link unknown
done
cd $model_zoo_home/tools
echo -e "\033[32m Convert benchmark_log done! \033[0m"


# Upload results to superset
database=mlu-validated
CNDB_DIR=$model_zoo_home/tools/cndb
if [[ ${database} != false ]]
then
    if [[ ${database} == "mlu-validated" ]]
    then
        db_file="mlu-validated-models.yaml"
    elif [[ ${database} == "mlu-demo" ]]
    then
        db_file="mlu-validated-demo.yaml"	    
    fi
    for file in $(ls ${RESULT_DIR}/*yaml); do
      cndb_submit \
        --db_file ${CNDB_DIR}/demo/config/${db_file}  \
        --load_file ${file} \
        --log-level INFO \
        --db_store_type save
    done
    echo -e "\033[32m Upload results done! You can check your training results on http://dataview.cambricon.com now! \033[0m"
else
    echo -e "\033[32m You choose to not upload the results to database! \033[0m"
fi

# Move performance yaml to logs file and delete other output files"
save_dir=$model_zoo_home/uploaded_results
mkdir -p "${save_dir}/results"
mv ${RESULT_DIR}/*yaml ${save_dir}/results/
rm ${RESULT_DIR} -fr