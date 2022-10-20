import argparse
import sys
import os
import yaml
import json
import logging
import re
import platform
import subprocess

from cndb.submit import submit
from cndb.params import CndbParams
from cndb.easy import get_mlu_name, dump_mlu_machine_info, dump_pt_info

logger = logging.getLogger(os.path.basename(__file__))


def parse_args():
    parser = argparse.ArgumentParser("tf dumper")
    parser.add_argument("-i", "--input", help="Input file path.")
    parser.add_argument("-o", "--outdir", help="Output YAML path.")
    parser.add_argument("--machine_name", help="Machine information name.")
    parser.add_argument("--framework", help="PyTorch version.")
    # parser.add_argument("--ct_commit_info", help="the commit id of master branch of catch.")
    parser.add_argument("--code_link", help="Code link.")

    args = parser.parse_args()
    if not args.input:
        parser.print_usage(sys.stderr)
        sys.exit(-1)
    return args

def run_cmd_info(cmd):
    try:
        if platform.system() == "Linux":
            all_info = subprocess.check_output(
                                cmd, stderr=subprocess.STDOUT, shell=True).strip()
            if type(all_info) is bytes:
                all_info = all_info.decode()
            return all_info
        else:
            logger.warning(
                "unsupported platform: {}".format(platform.system()))
    except Exception as e:
        logger.error("failed to run command: {}".format(cmd))
        raise e

def get_machine_name():
    try: 
        cmd = 'cat /etc/lsb-release | grep "DISTRIB_DESCRIPTION"'
        os_version = run_cmd_info(cmd)
        os_name = re.search(r"DISTRIB_DESCRIPTION=\"([\w\.\s]*)", os_version).group(1)
        return os_name
    except(subprocess.CalledProcessError):
        try:
            cmd = 'cat /etc/centos-release'
            os_version = run_cmd_info(cmd)
            os_name = re.search(r"CentOS([\sA-Za-z]*)([\d\.]*)", os_version).group(1)
            os_name = "CentOS-" + os_name
            return os_name
        except Exception as e:
            logger.error("unsupported os system: only ubuntu or centos.")
            raise e

class Reader:
    def __init__(self, hw_name, sw_name, code_link):
        # self.soft_info = json.loads(dump_pt_info(soft_name))
        self.sw_info = {'name': sw_name}
        self.hw_info = json.loads(dump_mlu_machine_info(hw_name))
        self.dev_name = get_mlu_name()
        self.code_link = code_link
        for key in self.hw_info.keys():
            if key in ['dev', 'cpu']:
                self.hw_info[key] = json.loads(self.hw_info[key])


    def read_line(self, line):
        """Read data from one line

        Data example:

            network:resnet50, Batch Size:256, device count:1, Precision:O0,\
            DPF mode:single, time_avg:0.511s, time_var:0.000178,\
            throughput(fps):501.2, device:MLU290, dataset:imageNet2012
        """
        data = {}
        performance = {}
        metrics = {}
        data["eval_type"] = "-"
        for field in line.split(","):
            key, value = field.strip().split(":")
            key = key.lower()
            value = value.strip()
            if key == "precision":
                data["train_type"] = value
            elif key == "eval_exec_mode":
                data["eval_type"] = value
            elif key == "dpf mode":
                data["dist_type"] = value
            elif key == "time_avg":
                performance["latency"] = value
            elif key == "time_var":
                performance["variance"] = value
            elif key == "fps" or "throughput" in key:
                performance["throughput"] = value
            elif key == "accuracy":
                metrics["accuracy"] = value
            elif key == "sw":
                metrics["sw"] = value
            elif key == "batch size":
                data["batch_size"] = int(value)
            elif key == "device count":
                data["dev_num"] = int(value)
            elif key == "network":
                data["model"] = value
            elif key == "driver":
                data["md5_val"] = value
            else:
                data[key] = value
        data["perf_data"] = performance
        if 'pytorch' in self.sw_info['name']:
            data["framework"] = "PyTorch"
        if 'tensorflow' in self.sw_info['name']:
            data["framework"] = "TensorFlow"
        if 'tensorflow2' in self.sw_info['name']:
            data["framework"] = "TensorFlow2"
        data["metric_data"] = metrics

        return data

    def dump(self, data, outfile):
        with open("./tools/soft_info.json") as f:
            sw_info_dict = json.load(f)
        # ct_version = re.search(r"(v[\d\.]*)", self.soft_info["name"]).group(1)
        ct_info = sw_info_dict[self.sw_info["name"]]

        self.sw_info["ctr_version"] = ct_info["ctr_version"]
        self.sw_info["framework_version"] = ct_info["framework_version"]
        self.sw_info["catch_version"] = ct_info["catch_version"]
        self.sw_info["release_date"] = ct_info["release_date"]

        data["code_link"] = self.code_link
        data["soft_info"] = self.sw_info
        data["hard_info"] = self.hw_info
        data["dev"] = data["device"] if "device" in data else self.dev_name
        data["save_file"] = outfile
        print(data)
        try:
            # cndb params check
            submit(CndbParams(data))
        except Exception as e:
            logger.error("failed to dump data to {}, due to {}".format(outfile, e))

    def read_and_dump(self, infile, outdir):
        with open(infile, "r") as fin:
            for line in fin.readlines():
                if line.strip() == "":
                    continue
                obj = self.read_line(line)
                if "model" not in obj:
                    logger.error("unknown record: {}".format(obj))
                    continue
                outfile = os.path.join(outdir, "{}_{}_{}_{}.yaml".format(
                    obj["model"],
                    obj["device"],
                    obj["dev_num"],
                    obj["batch_size"]
                    ))
                self.dump(obj, outfile)


if __name__ == "__main__":
    args = parse_args()
    # if args.pt_name is None:
    #     args.pt_name = get_pt_name() + "+" + args.ct_commit_info
    if args.machine_name is None:
        args.machine_name = get_machine_name()
    reader = Reader(args.machine_name, args.framework, args.code_link)
    reader.read_and_dump(args.input, args.outdir)

