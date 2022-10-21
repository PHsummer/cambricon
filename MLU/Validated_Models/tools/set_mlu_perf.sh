#!/bin/bash
#set -e

OS_NAME=NULL

function checkOs() {
  if [[ -f "/etc/lsb-release" ]];then
    OS_NAME=$(cat /etc/lsb-release | awk -F '=' '{if($1=="DISTRIB_ID") print $2}')
  elif [[ -f "/etc/redhat-release" ]];then
    OS_NAME="CentOS Linux"
  else
    echo -e "\033[31m ERROR: Only Support Ubuntu and CentOs.\033[0m"
    exit 1
  fi
}

function setCPUPerfMode() {
  if [ "$OS_NAME" == "Ubuntu" ]
  then
    installed_version=$(dpkg -l linux-tools-$(uname -r) | grep linux-tools-$(uname -r) | awk '{print $3}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}') bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]
    then
      apt-get install -y linux-tools-$(uname -r)
    fi
  elif [ "$OS_NAME" == "CentOS Linux" ]
  then
    installed_version=$(cpupower -v | awk '{if(NR==1) print $2}' | awk -F '.debug' '{print $1}')
    sys_version=$(uname -r | awk -F '-generic' '{print $1}')
    bool_match=$(echo $installed_version | grep $sys_version)
    if [ "$bool_match" == "" ]
    then
      yum install cpupowerutils
    fi
  else
    echo -e "\033[31m ERROR: Set Performance Mode Failed. Only Support Ubuntu and CentOs. \033[0m"
    exit 1
  fi
  performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
  if [ "$performance_mode" != "performance" ]
  then
    perf_cpu=$(cpupower -c all frequency-set -g performance)
    echo -e "\033[32m$perf_cpu \033[0m"
    # check performance mode
    performance_mode=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    if [ "$performance_mode" == "performance" ]
    then
      echo -e "\033[32m The CPU Performance Mode Enabled!\033[0m"
    else
      echo -e "\033[31m The CPU $performance_mode Mode Enabled! Please Check It.\033[0m"
      exit 1
    fi
  else
    echo -e "\033[32m The CPU $performance_mode Mode Enabled!\033[0m"
  fi
}

checkOs
setCPUPerfMode
