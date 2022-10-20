import os
import yaml
import logging

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import relationship, sessionmaker, scoped_session
from sqlalchemy_utils import database_exists, create_database, drop_database

from .model import (
    Base,
    ModelTable,
    SoftwareInfoTable,
    HardwareInfoTable,
    TrainResultTable,
)

logger = logging.getLogger(os.path.basename(__file__))


def get_db_session(params=None, config_file=None):
    if config_file != None:
        with open(config_file, "r") as fin:
            params = yaml.load(fin, Loader=yaml.FullLoader)
    if params is None:
        raise ValueError("DB parameters can not be none.")

    if "uri" in params:
        db_uri = params["uri"]
    else:
        schema = params["schema"]
        user = params["user"]
        passwd = params["passwd"]
        host = params["host"]
        port = params["port"]
        database = params["database"]
        db_uri = '{schema}://{username}:{passwd}@{host}:{port}/{database}'.format(
            schema=schema, username=user, passwd=passwd, host=host, port=port, database=database)
    if "postgresql" not in db_uri:
        raise ValueError("DB only support postgresql currently.")

    verbose = params["verbose"] != 0 if "verbose" in params else False
    rebuild = "rebuild" in params and params["rebuild"]

    logger.info("connect to: {}".format(db_uri))
    db_engine = create_engine(db_uri, echo=verbose)
    if not database_exists(db_engine.url):
        create_database(db_engine.url)

    if rebuild:
        logger.warning("drop database: {}".format(db_uri))
        Base.metadata.drop_all(db_engine)

    Base.metadata.create_all(db_engine, checkfirst=True)
    session_factory = scoped_session(sessionmaker(bind=db_engine))
    session = session_factory()

    return session


class DBHandler(object):
    def __init__(self, params):
        self.session = get_db_session(params=params)

    def store_result(self, data, mode="save"):
        required_keys = ("_model_name",)

        for key in required_keys:
            if key not in data:
                raise ValueError("Required key '{}' not in data".format(key))

        model_name = data.pop("_model_name")
        soft_info_name = data.pop("_soft_info_name")
        hard_info_name = data.pop("_hard_info_name")
        md5 = data["md5"]

        def q(cls, cond):
            ret = self.get_query(cls, cond)
            if not ret or len(ret) == 0:
                raise ValueError("{} from {} has no rows".format(cond, cls))
            return ret[0]

        def filter_col(data):
            # skip array columns, due to sqlachemy unsupported arrays '=' operation
            skip_cols = ["tags"]
            return {k: v for k, v in data.items() if k not in skip_cols}

        ret = self.session.query(TrainResultTable).filter_by(
            **filter_col(data)).first()
        if ret:
            logger.info("{} exist in {}".format(data, TrainResultTable))
            return ret

        ret_model = q(ModelTable, {"name": model_name})
        ret_soft = q(SoftwareInfoTable, {"name": soft_info_name})
        ret_hard = q(HardwareInfoTable, {"name": hard_info_name})

        if ret_model.metrics and data["metrics"]:
            for m in ret_model.metrics:
                if m not in data["metrics"]:
                    logger.warning("metric {} not exist in {} result, md5 {}".format(
                        m, data["name"], data["md5"]))
                    self.session.rollback()
                    return

        data["model_id"] = ret_model.id
        data["soft_info_id"] = ret_soft.id
        data["hard_info_id"] = ret_hard.id

        if mode == "save":
            rec = TrainResultTable(**data)
            self.session.add(rec)
            self.commit()
            logger.info("add record {} in {}".format(
                rec.id, rec.__table__.name))
        elif mode == "update":
            rec = self._get_or_store_by_cond(data, TrainResultTable, {
                                             "md5": data["md5"]}, True)
        else:
            rec = self._get_or_store_by_cond(data, TrainResultTable, {
                                             "md5": data["md5"]}, False)

        return rec

    def get_or_store_model(self, data, update=False):
        return self._get_or_store_by_cond(data, ModelTable, {"name": data["name"]}, update)

    def get_or_store_hard_info(self, data, update=False):
        return self._get_or_store_by_cond(data, HardwareInfoTable, {"name": data["name"]}, update)

    def get_or_store_soft_info(self, data, update=False):
        return self._get_or_store_by_cond(data, SoftwareInfoTable, {"name": data["name"]}, update)

    def get_query(self, cls, cond):
        return self.session.query(cls).filter_by(**cond).all()

    def get_model_names(self):
        names = self.session.query(ModelTable.name).all()
        return names

    def get_soft_info_names(self):
        return self._get_info_names(SoftwareInfoTable)

    def get_hard_info_names(self):
        return self._get_info_names(HardwareInfoTable)

    def _get_info_names(self, cls):
        names = self.session.query(cls.name).all()
        return names

    def _get_or_store_by_cond(self, data, cls, cond, update):
        ret = self.session.query(cls).filter_by(**cond).first()
        info = cls(**data)
        if ret:
            if update:
                info.id = ret.id
                self.session.merge(info)
                self.commit()
                logger.info("update record {} in table {}".format(
                    ret.id, ret.__table__.name))
            else:
                logger.warning("{} exist, skip".format(cond))
        else:
            self.session.add(info)
            self.commit()
            ret = info
            logger.info("update record {} in table {}".format(
                info.id, info.__table__.name))

        return ret

    def commit(self):
        self.session.commit()
