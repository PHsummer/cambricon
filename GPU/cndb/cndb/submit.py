import os
import logging
import json
import yaml

from .db import DBHandler
from .misc.util import md5
from .params import CndbParams

logger = logging.getLogger(os.path.basename(__file__))


def _not_none(key, data):
    if data is None:
        raise ValueError("{} is required".format(key))


def _not_none_warn(key, data):
    if data is None:
        logger.warning("{} is not set".format(key))


def _not_none_set(arr):
    for key, val in arr.items():
        _not_none(key, val)


def submit(args):
    p = CndbParams(args)

    _not_none_warn("--eval_type", p.eval_type)
    _not_none_warn("--train_type", p.train_type)
    _not_none_warn("--perf_data", p.perf_data)
    _not_none_warn("--metric_data", p.metric_data)
    _not_none_warn("--soft_file/--soft_info", p.soft_info)
    _not_none_warn("--hard_file/--hard_info", p.hard_info)

    if p.save_file:
        if p.dump():
            logger.info("save data in YAML file: {}".format(p.save_file))
    if p.db_info:
        db_handler = DBHandler(p.db_info)
    else:
        logger.info("--db_config/--db_file is not set, skip db operations.")
        return

    #model_name = p.model.lower()
    model_name = p.model
    _metric_data = {"name": model_name}
    if p.metric_data:
        metrics = p.metric_data.keys()
        _metric_data.update({"metrics": metrics})

    if p.soft_info:
        soft_info_name = p.soft_info["name"]
        p.soft_info.pop("name")
        soft_info_detail = p.soft_info
    else:
        soft_info_name = "unknown"
        soft_info_detail = ""

    if p.hard_info:
        hard_info_name = p.hard_info["name"]
        p.hard_info.pop("name")
        hard_info_detail = p.hard_info
    else:
        hard_info_name = "unknown"
        hard_info_detail = ""

    # save soft info
    db_soft_info = db_handler.get_or_store_soft_info(
        {"name": soft_info_name, "detail": soft_info_detail})

    # save hard info
    db_soft_info = db_handler.get_or_store_hard_info(
        {"name": hard_info_name, "detail": hard_info_detail})

    # model
    db_model = db_handler.get_or_store_model(data=_metric_data, update=True)
    db_handler.commit()

    data = {
        "_model_name": model_name,
        "_soft_info_name": soft_info_name,
        "_hard_info_name": hard_info_name,
        "framework": p.framework,
        "dataset": p.dataset,
        "batch_size": p.batch_size,
        "device": p.dev,
        "dev_num": p.dev_num,
        "train_exec_mode": p.train_type,
        "eval_exec_mode": p.eval_type,
        "dist_mode": p.dist_type,
        "hyper_params": p.hyper_params,
        "code_link": p.code_link,
        "tags": p.tags,
    }
    md5_val = p.md5_val
    if md5_val is None:
        data_str = json.dumps(data)
        md5_val = md5(data_str.encode("utf-8"))

    data.update({
        "md5": md5_val,
        "performance": p.perf_data,
        "metrics": p.metric_data,
        "extra": p.extra
    })

    db_handler.store_result(data, mode=p.db_store_type)
    db_handler.commit()
