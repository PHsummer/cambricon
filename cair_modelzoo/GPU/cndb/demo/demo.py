import yaml
import logging
import argparse
import copy

from cndb.db import DBHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo")


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Configuration YAML file, including DB, etc.")
    parser.add_argument("--model", help="Model YAML file.")
    parser.add_argument("--platform", help="Platform YAML file.")
    parser.add_argument("--result", help="Result YAML file")
    parser.add_argument("--show_names", dest="show_names", action="store_true")

    params = parser.parse_args()
    parser.set_defaults(show_names=False)

    return params


def yaml_loader(file):
    ret = None
    with open(file) as fin:
        ret = yaml.load(fin, Loader=yaml.SafeLoader)
    return ret


class DemoParser(object):
    def __init__(self):
        self.data = []

    def load(self, results):
        ret_templates = results["result_template"]
        for e in results["results"]:
            if "__template" in e:
                tname = e.pop("__template")
                if tname not in ret_templates:
                    logger.warning("{} not in templates, skip")
                    continue
                rec = copy.deepcopy(ret_templates[tname])
                rec.update(e)
                try:
                    self._valid(rec)
                    self.data.append(rec)
                except ValueError as e:
                    logger.error("skip record, due to {}".format(e))

    def iter(self):
        for rec in self.data:
            yield self.data.pop()

    def _valid(self, data):
        required_keys = [
            "md5", "framework", "_model_name", "_soft_info_name", "_hard_info_name",
            "dataset", "batch_size",
            "train_exec_mode", "eval_exec_mode", "dist_mode",
            "device", "dev_num",
            "code_link",
        ]

        for key in required_keys:
            if key not in data:
                raise ValueError("Required key '{}' not in data".format(key))

        if ("metrics" not in data) and ("throughtput" not in data):
            raise ValueError(
                "At least one of 'metrics' and 'throughtput' should be exist")

        return True


def run(p):
    db_config = yaml_loader(p.config)
    db = DBHandler(db_config)

    if p.show_names:
        print(db.get_model_names())
        print(db.get_soft_info_names())
        print(db.get_hard_info_names())
        return

    models = yaml_loader(p.model)
    platforms = yaml_loader(p.platform)
    results = yaml_loader(p.result)

    parser = DemoParser()

    for model in models:
        db.get_or_store_model(model)

    for info in platforms["hard_infos"]:
        db.get_or_store_hard_info(info)

    for info in platforms["soft_infos"]:
        db.get_or_store_soft_info(info)

    parser.load(results)
    cnt = 0
    for rec in parser.iter():
        db.store_result(rec)
        db.commit()


if __name__ == "__main__":
    p = get_params()
    run(p)
