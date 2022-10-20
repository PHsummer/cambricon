import argparse
import json
import logging
import os
import re

import yaml

logger = logging.getLogger(os.path.basename(__file__))


def _str_to_obj(data_str):
    try:
        obj = None
        if data_str and data_str != "":
            obj = json.loads(data_str)
        return obj
    except:
        raise ValueError(
            "failed to load json from string: \n{}".format(data_str))


def _str_to_arr(data_str):
    try:
        arr = None
        if data_str and data_str != "":
            arr = list(filter(
                lambda e: e != '', map(
                    lambda e: e.strip(), re.split(r",", data_str)
                )
            ))
        return arr
    except:
        raise ValueError(
            "failed to load json from string: \n{}".format(data_str))


def _file_to_obj(file):
    try:
        obj = None
        if file and file != "":
            with open(file, "r") as fin:
                obj = yaml.load(fin, Loader=yaml.FullLoader)
        return obj
    except:
        raise ValueError("failed to load yaml data from file: {}".format(file))


def _to_json(data, file):
    obj = None
    if data:
        obj = _str_to_obj(data)
    elif file:
        obj = _file_to_obj(file)
    return obj


def _info_valid(data):
    ret = "name" in data
    if not ret:
        logger.error("'name' should in data, current is {}".format(data))
    return ret


def _check(args):
    p = CndbParams(args)
    if not p.model:
        raise ValueError("{} is required".format("--model"))
    if not p.framework:
        raise ValueError("{} is required".format("--framework"))
    if not p.dataset:
        raise ValueError("{} is required".format("--dataset"))
    if not p.dev:
        raise ValueError("{} is required".format("--dev"))
    if p.dev_num <= 0:
        raise ValueError("--dev_num should be greater than 0.")
    if p.perf_data is None and p.metric_data is None:
        raise ValueError("Should set one of --perf_data or --metric_data")
    if p.soft_info and not _info_valid(p.soft_info):
        raise ValueError("--soft_info data is illegal")
    if p.hard_info and not _info_valid(p.hard_info):
        raise ValueError("--hard_info data is illegal")


def _add_arg_to_params(params, key, value, check_fn=None):
    check_value = value
    if check_fn:
        check_value = check_fn(value)
    if key in params:
        if check_value:
            params[key] = value
    else:
        params[key] = value


SOFT_INFO = "soft_info"
HARD_INFO = "hard_info"
DB_INFO = "db_info"
MODEL = "model"
BATCH_SIZE = "batch_size"
FRAMEWORK = "framework"
DATASET = "dataset"
DEV = "dev"
DEV_NUM = "dev_num"
EVAL_TYPE = "eval_type"
TRAIN_TYPE = "train_type"
DIST_TYPE = "dist_type"
PERF_DATA = "perf_data"
METRIC_DATA = "metric_data"
HYPER_PARAMS = "hyper_params"
CODE_LINK = "code_link"
MD5_VAL = "md5_val"
DB_STORE_TYPE = "db_store_type"
SAVE_FILE = "save_file"
TAGS = "tags"
EXTRA = "extra"

class CndbParams:
    def __init__(self, data):
        self.params = {}
        if type(data) is dict:
            self._read_data(data)
        elif type(data) is argparse.Namespace:
            self._read_args(data)
        elif type(data) is CndbParams:
            self.params = data.params
        else:
            raise ValueError("unknown type {}, try to use dict".format(data))

        for key, value in self.params.items():
            setattr(self, key, value)

    def _read_args(self, args):
        if args.load_file:
            self._load_file(args.load_file)

        _add_arg_to_params(self.params, SOFT_INFO, _to_json(
            args.soft_info, args.soft_file))
        _add_arg_to_params(self.params, HARD_INFO, _to_json(
            args.hard_info, args.hard_file))
        _add_arg_to_params(self.params, DB_INFO, _to_json(
            args.db_config, args.db_file))
        _add_arg_to_params(self.params, MODEL, args.model)
        _add_arg_to_params(self.params, BATCH_SIZE, args.batch_size)
        _add_arg_to_params(self.params, FRAMEWORK, args.framework)
        _add_arg_to_params(self.params, DATASET, args.dataset)
        _add_arg_to_params(self.params, DEV, args.dev)
        _add_arg_to_params(self.params, DEV_NUM,
                           args.dev_num, lambda e: e != 0)
        _add_arg_to_params(self.params, EVAL_TYPE, args.eval_type)
        _add_arg_to_params(self.params, TRAIN_TYPE, args.train_type)
        _add_arg_to_params(self.params, DIST_TYPE,
                           args.dist_type, lambda e: e != "single")
        _add_arg_to_params(self.params, PERF_DATA, _str_to_obj(args.perf_data))
        _add_arg_to_params(self.params, METRIC_DATA,
                           _str_to_obj(args.metric_data))
        _add_arg_to_params(self.params, HYPER_PARAMS,
                           _str_to_obj(args.hyper_params))
        _add_arg_to_params(self.params, CODE_LINK, args.code_link)
        _add_arg_to_params(self.params, MD5_VAL, args.md5_val)
        _add_arg_to_params(self.params, DB_STORE_TYPE, args.db_store_type)
        _add_arg_to_params(self.params, SAVE_FILE, args.save_file)
        _add_arg_to_params(self.params, TAGS, _str_to_arr(args.tags))
        _add_arg_to_params(self.params, EXTRA, _str_to_obj(args.extra))

    def _read_data(self, params):
        def _get_params(key):
            if key in params:
                return params[key]
            else:
                return None

        load_file = _get_params("load_file")
        if load_file:
            self._load_file(load_file)

        _add_arg_to_params(self.params, SOFT_INFO, _get_params("soft_info"))
        _add_arg_to_params(self.params, HARD_INFO, _get_params("hard_info"))
        _add_arg_to_params(self.params, DB_INFO, _get_params("db_info"))
        _add_arg_to_params(self.params, MODEL, _get_params("model"))
        _add_arg_to_params(self.params, BATCH_SIZE, _get_params("batch_size"))
        _add_arg_to_params(self.params, FRAMEWORK, _get_params("framework"))
        _add_arg_to_params(self.params, DATASET, _get_params("dataset"))
        _add_arg_to_params(self.params, DEV, _get_params("dev"))
        _add_arg_to_params(self.params, DEV_NUM, _get_params("dev_num"),
                           lambda e: e != 0)
        _add_arg_to_params(self.params, EVAL_TYPE, _get_params("eval_type"))
        _add_arg_to_params(self.params, TRAIN_TYPE, _get_params("train_type"))
        _add_arg_to_params(self.params, DIST_TYPE, _get_params("dist_type"),
                           lambda e: e != "single")
        _add_arg_to_params(self.params, PERF_DATA, _get_params("perf_data"))
        _add_arg_to_params(self.params, METRIC_DATA,
                           _get_params("metric_data"))
        _add_arg_to_params(self.params, HYPER_PARAMS,
                           _get_params("hyper_params"))
        _add_arg_to_params(self.params, CODE_LINK, _get_params("code_link"))
        _add_arg_to_params(self.params, MD5_VAL, _get_params("md5_val"))
        _add_arg_to_params(self.params, DB_STORE_TYPE,
                           _get_params("db_store_type"))
        _add_arg_to_params(self.params, SAVE_FILE, _get_params("save_file"))
        _add_arg_to_params(self.params, TAGS, _get_params("tags"))
        _add_arg_to_params(self.params, EXTRA, _get_params("extra"))

    def _load_file(self, file):
        try:
            with open(file, "r") as fin:
                params = yaml.load(fin, Loader=yaml.FullLoader)
            if "load_file" in params:
                params.pop("load_file")
            self._read_data(params)
        except Exception as e:
            logger.error(
                "failed to get params from {}, due to {}".format(file, e))

    def dump(self):
        dump_keys = [
            SOFT_INFO, HARD_INFO, MODEL, BATCH_SIZE, FRAMEWORK,
            DATASET, DEV, DEV_NUM, EVAL_TYPE, TRAIN_TYPE, DIST_TYPE,
            PERF_DATA, METRIC_DATA, HYPER_PARAMS, CODE_LINK, MD5_VAL, TAGS,EXTRA
        ]
        data = {}
        for key in dump_keys:
            _add_arg_to_params(data, key, self.params[key])
        try:
            with open(self.save_file, "w") as fout:
                yaml.dump(data, fout)
            return True
        except Exception as e:
            logger.error(
                "failed to dump params to {}, due to {}".format(self.save_file, e))
            return False
