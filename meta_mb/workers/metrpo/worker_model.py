import time, pickle
from meta_mb.logger import logger
from meta_mb.workers.worker_model_base import WorkerModelBase
from queue import Empty


class WorkerModel(WorkerModelBase):
    def __init__(self):
        super().__init__()
