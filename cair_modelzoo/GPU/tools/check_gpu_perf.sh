#!/bin/bash
#set -e

ignore_check=${1:-"no_ignore_check"}

function ignore_check() {
  if [ ${ignore_check} != "ignore_check" ]
  then 
    return 1
  else
    return 0
  fi
}

function print_log() {
  echo -e "\033[31m ERROR: Please execute the command: cd cair_modelzoo/GPU/tools/gpu_performance_check; sudo bash set_gpu_perf.sh\033[0m"
}


function check_mount() {
  dataset_path=$1
  dataset_first_path=$(echo $dataset_path | awk -F '/' '{print $2}')
  all_mount_paths=$(df -i | awk '{if(NR>1)print $6}')
  for path in $all_mount_paths
  do
    first_path=$(echo $path | awk -F '/' '{print $2}')
    if [[ $dataset_path =~ $path && $dataset_first_path == $first_path ]]
    then
      echo -e "\033[34m WARNING: The $dataset_path is mounted, it may degrade performance.\033[0m"
      # return 1
      break
    fi
  done
  return 0
}


function check_dataset_path() {
  check_mount $DATASETS_DIR
}

OS_NAME=NULL
function checkOs() {
  if [[ -f "/etc/lsb-release" ]];then
    OS_NAME=$(cat /etc/lsb-release | awk -F '=' '{if($1=="DISTRIB_ID") print $2}')
  elif [[ -f "/etc/redhat-release" ]];then
    OS_NAME="CentOS Linux"
  else
    echo -e "\033[31m ERROR: Only Support Ubuntu and CentOs.\033[0m"
  fi
}


function checkCPUInfo() {
    cpu_model=$(cat /proc/cpuinfo | awk -F ':' '{if ($1 ~ "model name") print $2}' | uniq)
    cpu_physical_core_num=$(cat /proc/cpuinfo |grep "physical id"|sort|uniq | wc -l)
    processor_num=$(cat /proc/cpuinfo | grep "processor" | wc -l)
    echo -e "\033[32m$cpu_model\033[0m"
    echo -e "\033[32m CPU Physical Core Nums: $cpu_physical_core_num\033[0m"
    echo -e "\033[32m CPU Processor Nums: $processor_num\033[0m"
}

function FindCPUProcess() {
  info=$(top -n 1 | head -n 20 | awk '{if (NR>7 && NF==14 && $10>5) print $2, $10, $13; else if (NR>7 && NF==13 && $9>5) print $1, $9, $12}' | awk '{if ($3!="top") printf(" ERROR: The PID: %10s, COMMAND: %20s, Please Kill It.\n", $1, $3)}')
  if [ "$info" != "" ]
  then
    echo -e "\033[31m$info \033[0m"
    ignore_check 
#    if [ ${ignore_check} != "ignore_check" ]
#    then
#      exit 1 
#    fi
else
    echo -e "\033[32m No Programs Occupied CPUs!\033[0m"
  fi
}

function FindGPUPID() {
  line_num=$(nvidia-smi | awk '{if($0 ~ "PID") print NR}')
  pid_info=$(nvidia-smi | awk -v line=$line_num '{a=line;if(NR>a && NF == 9) printf(" ERROR: The PID %s Running On GPU, Please Kill It.\n", $5)}')
  if [ "$pid_info" != "" ]
  then
    echo -e "\033[31m$pid_info \033[0m"
    ignore_check
  fi
}

function setEnv() {
  if [ -z "${CONT}" ]
  then
    echo -e "\033[31m ERROR : Docker image name is not set\033[0m"
  fi
  if [ -z "${DATASETS}"  ]
  then
    echo -e "\033[31m ERROR : Dataset name is not set\033[0m"
  fi
  if [ -z "${DATASETS_DIR}"  ] 
  then
    echo -e "\033[31m ERROR : Dataset dir is not set\033[0m"
  fi
  if [ -z "${CNDB_DIR}"  ]
  then
    echo -e "\033[31m ERROR : CNDB dir is not set\033[0m"
  fi
  if [ -z "${CODE_LINK}"  ] 
  then
    echo -e "\033[31m ERROR : Code link is not set\033[0m"
  fi    
  if [ -z "${RUN_MODEL_FILE}"  ] 
  then
    echo -e "\033[31m ERROR : Run model script is not set\033[0m"
  fi  
  if [ -z "${DOCKERFILE}"  ] 
  then
    echo -e "\033[31m ERROR : Dockerfile is not set\033[0m"
  fi  
}

function checkCPUPerfMode() {
  if [ "$OS_NAME" == "Ubuntu" ]
  then
    installed_version=$(dpkg -l linux-tools-$(uname -r) | grep linux-tools-$(uname -r) | awk '{print $3}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}')
    bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]
    then
      echo -e "\033[31m ERROR: Mismatch linux-tools-$(uname -r)\033[0m"
      print_log
    fi
  elif [ "$OS_NAME" == "CentOS Linux" ]
  then
    installed_version=$(cpupower -v | awk '{if(NR==1) print $2}' | awk -F '.debug' '{print $1}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}')
    bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]
    then
      echo -e "\033[31m ERROR: Mismatch cpupower\033[0m"
      print_log
    fi
  else
    echo -e "\033[31m ERROR: Set Performance Mode Failed. Only Support Ubuntu and CentOs. \033[0m"
  fi
  performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
  if [ "$performance_mode" != "performance" ]
  then
    echo -e "\033[31m ERROR: The CPU $performance_mode Mode Enabled!\033[0m"
    print_log
    ignore_check
  else
    echo -e "\033[32m The CPU Performance Mode Enabled!\033[0m"
  fi
}

function irqbalanceCheck() {
  irqbalance_min_version="1.5.0"
  irq_v=$(irqbalance --version)
  if [ "irq_v" != "" ]
  then 
    irq_version=$(irqbalance --version | awk '{print $3}')
  fi
  if test "$(echo ${irq_version} ${irqbalance_min_version} | tr " " "\n" | sort -V | head -n 1)" != "1.5.0"
  then
    echo -e "\033[31m ERROR : irqbalance minimal version is 1.5.0, please upgrade irqbalance\033[0m"
    ignore_check
  else
      return 0
  fi
  if [ "$OS_NAME" == "Ubuntu" ]
  then
      irqB_version=$(dpkg -l | grep irqbalance | awk '{print($3)}' | cut -d "-" -f 1)
      if [ ! -n "$irqB_version" ]
      then
        echo -e "\033[31m ERROR : irqbalance is not installed\033[0m"
        ignore_check
      elif test "$(echo ${irqB_version} ${irqbalance_min_version} | tr " " "\n" | sort -V | head -n 1)" != "1.5.0"
      then
        echo -e "\033[31m ERROR : irqbalance minimal version is 1.5.0, please upgrade irqbalance\033[0m"
        ignore_check
      fi
  elif [ "$OS_NAME" == "CentOS Linux" ]
  then
      irqB_version=$(yum list installed | grep irqbalance | awk '{print($2)}' | cut -b 3-7)
      if [ ! -n "$irqB_version" ]
      then
        echo -e "\033[31m ERROR : irqbalance is not installed\033[0m"
        ignore_check
      elif test "$(echo ${irqB_version} ${irqbalance_min_version} | tr " " "\n" | sort -V | head -n 1)" != "1.5.0"
      then
        echo -e "\033[31m ERROR : irqbalance minimal version is 1.5.0, please upgrade irqbalance\033[0m"
        ignore_check
      fi    
  fi
}

function ACSCheck() {
    acs=$(lspci -vvv | grep -i acsctl)
    if [ -n "$CONT_FIND" ]
    then
    echo -e "\033[31m ERROR : ACS is not closed, please close ACS\033[0m"
    ignore_check
    fi
}

check_dataset_path
setEnv
checkOs
FindCPUProcess
[[ $? -eq 0 ]] && echo -e "\033[32m Check CPU Process Success! \033[0m" || exit
FindGPUPID
[[ $? -eq 0 ]] && echo -e "\033[32m Check GPU Process Success! \033[0m" || exit
checkCPUPerfMode
[[ $? -eq 0 ]] && echo -e "\033[32m Check CPU Performance Mode Success! \033[0m" || exit
irqbalanceCheck
[[ $? -eq 0 ]] && echo -e "\033[32m Check irqbalance Success! \033[0m" || exit
ACSCheck
[[ $? -eq 0 ]] && echo -e "\033[32m Check ACS Success! \033[0m" || exit
















