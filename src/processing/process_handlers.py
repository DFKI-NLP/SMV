import multiprocessing
from dataclasses import dataclass
from typing import Union, List

import psutil

from src.dataloader import Verbalizer
import src.search_methods.spans as s
import src.search_methods.filters as f
import src.search_methods.tools as t

import src.processing.shared_methods as sm


@dataclass(init=True)
class TaskBase:
    TaskName: str
    RequiredTasks: List[str]
    RequiredRamPerProcess:  int  # in GBytes f.e. 1 = 1024*1024*1024 bytes
    DesiredProcesses: int
    TaskIndex: int
    Task: staticmethod
    done = False
    associated_processes = []

    def __str__(self):
        return self.TaskName


def conv_task():
    return TaskBase("ConvSearch", None, 3, 4, 0, sm.convolution_search)


def span_task():
    return TaskBase("SpanSearch", None, 3, 1, 1, sm.span_search)


def concat_task():
    return TaskBase("ConcatSearch", ["ConvSearch", "SpanSearch"], 3, 2, 2, sm.concatenation_search)


########################################################################################################################
########################################################################################################################
########################################################################################################################


class ProcessHandler:
    def __init__(self, loader: Verbalizer, tasks: List[TaskBase]):

        self.root = loader
        self.tasks = self.order_tasks(tasks)
        self.fulfilled_tasks = []

    @staticmethod
    def order_tasks(tasks: List[TaskBase]):  # simple bubble sort as task length is very small
        for _ in range(len(tasks)-1):
            for i in range(len(tasks)-1):
                if tasks[i].TaskIndex > tasks[i+1].TaskIndex:
                    tasks[i], tasks[i+1] = tasks[i+1], tasks[i]
        return tasks

    @staticmethod
    def get_available_ram():
        return psutil.virtual_memory().available / (1024**3)

    def check_requirements(self, task:TaskBase):
        req = True
        if task.RequiredTasks:
            for i in task.RequiredTasks:
                if i not in self.fulfilled_tasks:
                    req = False
        return req

    def mainloop(self):
        pass

    def start_task(self, task: TaskBase):
        if self.get_available_ram() > task.RequiredRamPerProcess:
            if self.check_requirements(task):
                pass




