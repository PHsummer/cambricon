import argparse
import logging
import os
import sys

from .params import _check
from .submit import _not_none_set, submit

LOG_LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'FATAL']


def parse_args():
    parser = argparse.ArgumentParser("cndb parameters")
    parser.add_argument("--model", dest="model",
                        help="Training Model.")
    parser.add_argument("--batch_size", dest="batch_size", type=int,
                        help="Batch size per device.")
    parser.add_argument("--framework", dest="framework",
                        help="Framework.")
    parser.add_argument("--dataset", dest="dataset",
                        help="Framework")
    parser.add_argument("--dev", dest="dev",
                        help="Device type.")
    parser.add_argument("--dev_num", dest="dev_num", default=0, type=int,
                        help="Device number.")
    parser.add_argument("--train_type", dest="train_type",
                        help="Training type.")
    parser.add_argument("--eval_type", dest="eval_type",
                        help="Evaluation type")
    parser.add_argument("--dist_type", dest="dist_type", default="single",
                        help="Distributed type, default single.")
    parser.add_argument("--perf_data", dest="perf_data",
                        help="Performance Json data.")
    parser.add_argument("--metric_data", dest="metric_data",
                        help="Metircs Json data.")
    parser.add_argument("--hyper_params", dest="hyper_params",
                        help="Hyper-parameter.")
    parser.add_argument("--soft_info", dest="soft_info",
                        help="Software information (optional).")
    parser.add_argument("--soft_file", dest="soft_file",
                        help="Software information YAML file (optional). Skip when soft_info exist.")
    parser.add_argument("--hard_info", dest="hard_info",
                        help="Hardware information (optional).")
    parser.add_argument("--hard_file", dest="hard_file",
                        help="Hardware information YAML file (optional). Skip when hard_info exist.")
    parser.add_argument("--code_link", dest="code_link",
                        help="Code link.")
    parser.add_argument("--db_config", dest="db_config",
                        help="Database configuration in Json format.")
    parser.add_argument("--db_file", dest="db_file",
                        help="Database config file in YAML. Skip when db_config exist.")
    parser.add_argument("--md5", dest="md5_val",
                        help="md5 for one record.")
    parser.add_argument("--tags", dest="tags",
                        help="tags which are splitted by comma.")
    parser.add_argument("--db_store_type", dest="db_store_type", default="save",
                        help="Database update mode, include "
                        "'save': insert a record whether md5 is duplicated, "
                        "'update': update the first found reocrd when md5 exists, otherwise insert a new record, "
                        "default skip when md5 exists, otherwise insert a new record.")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="WARNING",
                        help="Minimum level to log to stderr. (default: WARNING).")
    parser.add_argument("-f", "--data_file", dest="file",
                        help="YAML file of all the information")
    parser.add_argument("--load_file", dest="load_file",
                        help="Load all information from YAML file.")
    parser.add_argument("--save_file", dest="save_file",
                        help="Save YAML file of all the information.")

    args = parser.parse_args()

    try:
        _check(args)
    except ValueError as e:
        parser.print_usage(sys.stderr)
        print("{}: error: {}".format(os.path.basename(sys.argv[0]), e))
        sys.exit(1)
    return args


def run_commandline():
    args = parse_args()

    if args.log_level:
        FORMAT = "[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s"
        logging.addLevelName(logging.NOTSET, 'TRACE')
        logging.basicConfig(
            level=logging.getLevelName(args.log_level),
            format=FORMAT,
            datefmt='%Y-%m-%d:%H:%M:%S'
        )

    submit(args)


if __name__ == '__main__':
    run_commandline()
