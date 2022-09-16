import multiprocessing as mp
import multiprocessing.shared_memory as smm
import sys
from array import array

import numpy as np
from dataclasses import dataclass
from typing import Union, List, Tuple

import psutil

from src.dataloader import Verbalizer
import src.search_methods.searches as f
import src.search_methods.tools as t

import src.processing.shared_methods as sm


@dataclass(init=True)
class TaskBase:
    TaskName: str
    RequiredTasks: List[str]
    RequiredRamPerProcess:  int  # in GBytes f.e. 1 = 1024*1024*1024 bytes
    DesiredProcesses: int
    TaskIndex: int
    Task: any  # preferably a method but anything callable works
    started = False
    done = False
    associated_processes = []  # maybe idk
    associated_pipes = []
    dones = []

    def __str__(self) -> str:
        return self.TaskName

    def __repr__(self) -> str:
        return f"Task {self.TaskName}"

def conv_task() -> TaskBase:
    return TaskBase("convolution search", [], 3, 4, 0, sm.convolution_search)


def span_task() -> TaskBase:
    return TaskBase("span search", [], 3, 1, 1, sm.span_search)


def concat_task() -> TaskBase:
    return TaskBase("concatenation search", ["convolution search", "span search"], 3, 2, 2, sm.concatenation_search)


########################################################################################################################
########################################################################################################################
########################################################################################################################


class ProcessHandler:
    # constants

    def __init__(self, loader: Verbalizer,
                 tasks: List[TaskBase],
                 samples: dict,
                 maxram: float = -1):

        self.root = loader
        self.tasks = self.order_tasks(tasks)
        self.samples = samples
        self.maxram = maxram

        self.fulfilled_tasks = []
        self.orders_and_searches = None

    @staticmethod
    def order_tasks(tasks: List[TaskBase]) -> List[TaskBase]:  # simple bubble sort as task length is very small
        for _ in range(len(tasks)-1):
            for i in range(len(tasks)-1):
                if tasks[i].TaskIndex > tasks[i+1].TaskIndex:
                    tasks[i], tasks[i+1] = tasks[i+1], tasks[i]
        return tasks

    @staticmethod
    def equalsplit_data(data: np.array, pieces: int) -> list:  # https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
        """Yield successive pieces-sized chunks from data."""
        for i in range(0, len(data), pieces):
            yield data[i:i + pieces]

    def get_available_ram(self) -> float:
        return min(psutil.virtual_memory().available / (1024**3), self.maxram if self.maxram >= 0 else sys.maxsize)

    def get_args(self, task: TaskBase) -> dict:
        if task.TaskName == "ConvSearch":
            return {"sgn": self.root.sgn,
                    "sample_array": self.samples,
                    "len_filters": self.root.len_filters,
                    "metric": self.root.metric,
                    "shared_search": None,
                    "shared_order": None}

        if task.TaskName == "SpanSearch":
            return {"sgn": self.root.sgn,
                    "sample_array": self.samples,
                    "len_filters": self.root.len_filters,
                    "metric": self.root.metric,
                    "shared_search": None,
                    "shared_order": None}

        if task.TaskName == "ConcatSearch":
            return {}

    def check_requirements(self, task: TaskBase) -> bool:
        req = True
        if task.RequiredTasks:
            for i in task.RequiredTasks:
                if i not in self.fulfilled_tasks:
                    req = False
        return req

    def __call__(self, *args, **kwargs) -> dict:
        orders_and_searches = {}
        for task in self.tasks:
            if not task.done:
                for process in range(len(task.associated_processes)):
                    task.associated_processes[process].join(timeout=0.010)  # 10ms
                    if task.associated_processes[process].is_alive:
                        continue
                    else:
                        _ = task.associated_pipes[process].recv()


        return orders_and_searches

    def start_task(self, task: TaskBase) -> None:
        processes = []
        pipes = []
        if self.get_available_ram() > task.RequiredRamPerProcess:
            if self.check_requirements(task):
                num_procs = int(min(self.get_available_ram()/task.RequiredRamPerProcess, task.DesiredProcesses))
                ind_slicer = self.equalsplit_data(list(self.samples.keys()), num_procs)
                for i in range(num_procs):
                    par, child = mp.Pipe()
                    data = {}
                    for keys in next(ind_slicer):
                        for key in keys:
                            data[key] = {"input_ids": self.samples[key]["input_ids"],
                                         "attributions": self.samples[key]["attributions"]}
                    processes.append(mp.Process(target=task.Task, args=(self.root.sgn,
                                                                        data,
                                                                        self.root.len_filters,
                                                                        self.root.metric,
                                                                        child),
                                                daemon=True)
                                     )
                    pipes.append(par)

        task.associated_processes = processes
        task.associated_pipes = pipes
        for i in processes:
            i.start()
        task.started = True




