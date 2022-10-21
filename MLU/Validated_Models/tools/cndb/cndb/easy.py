import os
import re
import sys
import platform
import subprocess
import yaml
import json
import logging

from .misc.util import md5

logger = logging.getLogger(os.path.basename(__file__))


def _get_cmd_info(command):
    try:
        if platform.system() == "Linux":
            all_info = subprocess.check_output(
                command, stderr=subprocess.STDOUT, shell=True).strip()
            if type(all_info) is bytes:
                all_info = all_info.decode()
            return all_info
        else:
            logger.warning(
                "unsupported platform: {}".format(platform.system()))
    except Exception as e:
        logger.error("failed to run command: {}".format(command))

    return ""


def _get_cpu_info():
    all_info = _get_cmd_info("lscpu")

    cpu_info = {}
    cpu_name = ""
    cpu_phy_count = 0
    cpu_logic_count = 0
    for line in all_info.split("\n"):
        if "Model name" in line:
            cpu_name = re.search(r"^Model name:\s*(.*)", line).group(1)
            # cpu_name = re.sub("^Model name:\s+(\d+)", "", line, 1).strip()
        elif line.startswith("Socket(s)"):
            cpu_phy_count = re.search(r"^Socket\(s\):\s*(\d+)", line).group(1)
        elif line.startswith("CPU(s)"):
            cpu_logic_count = re.search(r"^CPU\(s\):\s*(\d+)", line).group(1)
    cpu_info = {
        "name": cpu_name,
        "sockets": cpu_phy_count,
        "cores": cpu_logic_count
    }
    return json.dumps(cpu_info)


def _get_mlu_info():
    all_info = _get_cmd_info("cat /proc/driver/cambricon/mlus/*/information")
    dev_info = {}

    def obj_add(name, obj):
        if name in obj:
            obj[name] += 1
        else:
            obj[name] = 1

    try:
        for line in all_info.split("\n"):
            if "Device name" in line:
                dev_name = re.sub(".*Device name.*:", "", line, 1).strip()
                obj_add(dev_name, dev_info)
            elif "LnkCap" in line:
                lnkcap = re.sub(".*LnkCap.*:", "", line, 1).strip()
                obj_add(lnkcap, dev_info)
    except Exception as e:
        logger.error("failed to get device information")
    return json.dumps(dev_info)


def _get_host_mem():
    all_info = _get_cmd_info("free -g")
    try:
        for line in all_info.split("\n"):
            if "Mem" in line:
                return "{} GB".format(re.split(r"\s+", line)[1])
    except:
        logger.error("failed to get host memory")
    return ""


def _dump(msg, file):
    if file:
        with open(file, "w") as fout:
            yaml.dump(msg, fout)
    return json.dumps(msg)


def get_mlu_name():
    """MLU device name

    Get device name from '/proc/driver/cambricon/mlus/*/information'

    Returns:
        Device name or 'unknown'
    """
    all_info = _get_cmd_info("cat /proc/driver/cambricon/mlus/*/information")
    for line in all_info.split("\n"):
        if "Device name" in line:
            return re.sub("Device name.*:", "", line, 1).strip()
    return "unknown"


def dump_mlu_machine_info(name=None, file=None):
    """Dump MLU Machine information to STDOUT or file in YAML format.

    Get machine information include

    - name: machine name
    - cpu: cpu information, include socket number, logical core number
    - mem: total memory
    - dev: mlu information, include mlu name, number, pcie capability

    Args:
        name: information name. Use platform node name if None.
        file: output filename. Print to stdout when None.

    Returns:
        None
    """
    info = {
        "name": name if name else platform.node(),
        "cpu": _get_cpu_info(),
        "mem": _get_host_mem(),
        "dev": _get_mlu_info()
    }

    return _dump(info, file)


def dump_tf_info(name, file=None, hash_name=False):
    """Dump TensorFlow and dependent libraries version to STDOUT or file in YAML format.

    Get TensorFlow and Cambricon Neuware Libraries version information, include

    - "tf": TensorFlow version
    - "camb_tf": Cambricon Neuware version
    - "driver": MLU Driver version
    - "cnrt": CNRT version
    - "cnnl": CNNL version
    - "cnml": CNML version

    Args:
        name: information name
        file: output filename. Print to stdout when None.

    Returns:
        None
    """
    info = {}
    try:
        tf_info = _get_cmd_info(
            "python -c '"
            "import tensorflow as tf;"
            "print(\"__CNDB__TF_VERSION: {}\".format(tf.__version__));"
            "print(\"__CNDB__CAMB_TF_VERSION: {}\".format(tf.version.GIT_VERSION));"
            "tf.config.experimental_list_devices() if tf.__version__.startswith(\"1\") "
            "else tf.config.list_logical_devices();"
            "'"
        )
        cnmon_info = _get_cmd_info("cnmon info -c 0 -t")

        driver = ""
        cnrt = ""
        cnnl = ""
        cndrv = ""
        camb_tf = ""
        camb_tf_commit = ""
        tf = ""
        for line in tf_info.split("\n"):
            if "Current library versions" in line:
                cnrt = re.search(r"CNRT: ([\d\.]*)", line).group(1)
                cnnl = re.search(r"CNNL: ([\d\.]*)", line).group(1)
                cndrv = re.search(r"CNDrv: ([\d\.]*)", line).group(1)
            elif "Cambricon TensorFlow version" in line:
                camb_tf = re.search(r"version: ([\d\.]*)", line).group(1)
            elif "__CNDB__TF_VERSION" in line:
                tf = re.search(r"__CNDB__TF_VERSION: ([\d\.]*)", line).group(1)
            elif "__CNDB__CAMB_TF_VERSION" in line:
                camb_tf_commit = re.search(
                    r"__CNDB__CAMB_TF_VERSION: ([\w\-\d\.]*)", line).group(1)
        for line in cnmon_info.split("\n"):
            if "Driver" in line:
                driver = re.search(r".*Driver.*?([v\d\.]+)", line).group(1)

        info = {
            "tf": tf,
            "camb_tf": camb_tf,
            "camb_tf_commit": camb_tf_commit,
            "driver": driver,
            "cnrt": cnrt,
            "cnnl": cnnl,
            "cndrv": cndrv,
        }

        json_str = json.dumps(info)
        name_md5 = md5(json_str.encode("utf8"))[:6]
        if name == "" or name is None:
            name = name_md5
        elif hash_name:
            name = name + "-" + name_md5
        info["name"] = name
    except Exception as e:
        err_msg = "failed to get tf information, due to {}".format(e)
        info = {"name": "error", "msg": err_msg}
        logger.error(err_msg)

    return _dump(info, file)


def dump_pt_info(name, file=None, hash_name=False):
    """Dump Pytorch and dependent libraries version to STDOUT or file in YAML format.

    Get Pytorch and Cambricon Neuware Libraries version information, include

    - "pt": TensorFlow version

    Args:
        name: information name
        file: output filename. Print to stdout when None.

    Returns:
        None
    """
    info = {}
    try:
        all_info = _get_cmd_info(
            "python -c '"
            "import torch_mlu;"
            "print(\"__CNDB__PT_VERSION: {}\".format(torch_mlu.torch.__version__));"
            "'"
        )
        pt = ""
        for line in all_info.split("\n"):
            if "__CNDB__PT_VERSION" in line:
                pt = re.search(
                    r"__CNDB__PT_VERSION: ([\d\.\+\w+]*)", line).group(1)
        info = {
            "pt": pt,
        }
        json_str = json.dumps(info)
        name_md5 = md5(json_str.encode("utf8"))[:6]
        if name == "" or name is None:
            name = name_md5
        elif hash_name:
            name = name + "-" + name_md5
        info["name"] = name
    except Exception as e:
        err_msg = "failed to get pt information, due to {}".format(e)
        info = {"name": "error", "msg": err_msg}
        logger.error(err_msg)

    return _dump(info, file)
