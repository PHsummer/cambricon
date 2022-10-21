from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Text, Float, Enum, DateTime, Boolean, ARRAY,
    ForeignKey,
    UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ModelTable(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)

    metrics = Column(ARRAY(String))
    categories = Column(ARRAY(String))


class SoftwareInfoTable(Base):
    __tablename__ = "software_infos"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    detail = Column(JSONB)


class HardwareInfoTable(Base):
    __tablename__ = "hardware_infos"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    detail = Column(JSONB)


class TrainResultTable(Base):
    __tablename__ = "train_results"

    id = Column(Integer, primary_key=True)
    md5 = Column(String)
    framework = Column(String)
    dataset = Column(String)
    batch_size = Column(Integer)  # total batch size
    device = Column(String)
    dev_num = Column(Integer)
    train_exec_mode = Column(String)
    eval_exec_mode = Column(String)
    dist_mode = Column(String)
    cmd = Column(Text)
    code_link = Column(String)
    metrics = Column(JSONB)
    performance = Column(JSONB)
    hyper_params = Column(JSONB)

    tags = Column(ARRAY(String, dimensions=1))
    # we don't know what information in future, use this to store some data
    extra = Column(JSONB)

    visible = Column(Boolean, default=False)
    create_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    update_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    model_id = Column(Integer, ForeignKey("models.id"))
    hard_info_id = Column(Integer, ForeignKey("hardware_infos.id"))
    soft_info_id = Column(Integer, ForeignKey("software_infos.id"))
