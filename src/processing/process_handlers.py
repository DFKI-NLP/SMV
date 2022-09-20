import multiprocessing as mp
import sys
from numba import jit
import time

import numpy as np
from dataclasses import dataclass
from typing import Union, List, Tuple

import psutil

from src.dataloader import Verbalizer
import src.search_methods.searches as f
import src.search_methods.tools as t

import src.processing.shared_value_searches as sh


@dataclass(init=True)
class Worker:
    process: mp.Process
    parent_pipe: mp.connection
    child_pipe: mp.connection
    needs_update: bool


@dataclass(init=True)
class WorkerManager:
    TaskName: str
    RequiredTasks: List[str]
    RequiredRamPerProcess:  int  # in GBytes f.e. 1 = 1024*1024*1024 bytes
    DesiredProcesses: int
    TaskIndex: int
    Task: any  # preferably a method but anything callable works
    TaskID = None
    started = False
    active = True
    done = False
    workers = []  # List[Worker]

    iterator = None  # generator object

    def __str__(self) -> str:
        return self.TaskName

    def __repr__(self) -> str:
        return f"Task {self.TaskName}"

    def start(self) -> None:
        for worker in self.workers:
            worker.process.start()
        self.started = True

    def get(self) -> Tuple[List[Tuple[dict, list, bool]], List[int]]:
        result = []
        workers_without_work = []

        for worker in range(len(self.workers)):
            if self.workers[worker].parent_pipe.poll():
                result.append(self.workers[worker].parent_pipe.recv())
                self.workers[worker].needs_update = True
            else:
                if self.workers[worker].needs_update:
                    workers_without_work.append(worker)
        return result, workers_without_work

    def set(self, index: int, data: any) -> None:
        self.workers[index].needs_update = False
        self.workers[index].parent_pipe.send(data)

    def kill(self, index) -> None:
        self.workers[index].parent_pipe.close()
        self.workers[index].child_pipe.close()
        self.workers[index].process.terminate()
        self.workers[index].process.kill()


def conv_manager() -> WorkerManager:
    return WorkerManager("convolution search", [], .3, 4, 0, sh.shared_memory_convsearch)


def span_manager() -> WorkerManager:
    return WorkerManager("span search", [], .3, 1, 1, sh.shared_memory_spansearch)


def concat_manager() -> WorkerManager:
    return WorkerManager("concatenation search", ["convolution search", "span search"], 3, 2, 2, sh.shared_memory_compare_searches)


########################################################################################################################
########################################################################################################################
########################################################################################################################


class ProcessHandler:
    # constants

    def __init__(self, loader: Verbalizer,
                 managers: List[WorkerManager],
                 samples: dict,
                 maxram: float = -1):

        self.root = loader
        self.managers = self.order_tasks(managers)
        self.samples = samples
        self.sample_keys = list(samples.keys())
        self.maxram = maxram

        self.working_managers = []
        self.fulfilled_tasks = [False for i in range(len(managers))]
        self.orders_and_searches = {}
        self.explanations = {}

    @staticmethod
    def order_tasks(managers: List[WorkerManager]) -> List[WorkerManager]:  # simple bubble sort as task length is very small
        for _ in range(len(managers) - 1):
            for i in range(len(managers) - 1):
                if managers[i].TaskIndex > managers[i + 1].TaskIndex:
                    managers[i], managers[i + 1] = managers[i + 1], managers[i]

        for i in range(len(managers)):
            managers[i].TaskID = i

        return managers

    def get_available_ram(self) -> float:
        return min(psutil.virtual_memory().available / (1024**3), self.maxram if self.maxram >= 0 else sys.maxsize)

    def get_args(self, task: WorkerManager) -> dict:
        return {}

    @staticmethod
    def iterator(listlike) -> any:
        for i in range(len(listlike)):
            yield listlike[i]

    def check_requirements(self, manager: WorkerManager) -> bool:
        req = True
        if manager.RequiredTasks:
            for i in manager.RequiredTasks:
                if i not in self.fulfilled_tasks:
                    req = False
        return req

    def __call__(self, *args, **kwargs) -> dict:
        for manager in self.managers:
            self.start_manager(manager)
            self.orders_and_searches[manager.TaskName] = {}
            self.explanations[manager.TaskName] = {}
        print("Started manager(s)")

        working = True
        print("now checking")
        ti = time.time()
        while working:
            for manager in self.working_managers:
                working = False
                self.check_manager(manager)
                if manager.active:
                    working = True
        print(time.time() - ti)
        return self.orders_and_searches

    def generate_worker(self, manager: WorkerManager) -> Worker:
        parent, child = mp.Pipe()
        process = mp.Process(target=manager.Task, args=(self.root.sgn,
                                                        self.root.len_filters,
                                                        self.root.metric,
                                                        child),
                             daemon=True,
                             name=manager.TaskName)
        return Worker(process, parent, child, True)

    def check_manager(self, manager: WorkerManager) -> None:
        result, workers_without_work = manager.get()
        for entry in result:
            self.orders_and_searches[manager.TaskName][entry[2]] = entry[0]
            self.explanations[manager.TaskName][entry[2]] = entry[1]
        for worker in workers_without_work:
            try:
                if manager.active:
                    key = next(manager.iterator)
                    manager.set(worker, (self.samples[key], key))
                else:
                    manager.kill(worker)
            except StopIteration:
                manager.active = False

    def start_manager(self, manager: WorkerManager) -> None:
        worker_objects = []
        if self.check_requirements(manager):
            available_ram = self.get_available_ram()
            if available_ram > manager.RequiredRamPerProcess:
                num_workers = min(int(available_ram/manager.RequiredRamPerProcess), manager.DesiredProcesses)
                for i in range(num_workers):
                    worker_objects.append(self.generate_worker(manager))
                manager.workers = worker_objects
                manager.iterator = self.iterator(self.sample_keys)
                manager.start()
                self.working_managers.append(manager)
            else:
                raise MemoryError("Not enough memory to start at least 1 process")

