import multiprocessing as mp
import sys
import warnings
import time

from dataclasses import dataclass
from typing import Union, List, Tuple
import psutil
import src.processing.processing_tools as sh


@dataclass(init=True)
class Worker:
    process: mp.Process
    parent_pipe: mp.connection
    child_pipe: mp.connection
    needs_update: bool

    alive = True


@dataclass(init=True)
class WorkerManager:
    TaskName: str
    RequiredTasks: List[str]
    RequiredRamPerProcess:  float or int  # in GBytes f.e. 1 = 1024*1024*1024 bytes
    DesiredProcesses: int
    TaskIndex: int
    Task: any  # preferably a method but anything callable works
    TaskID = None
    started = False
    active = True
    done = False
    workers = []  # List[Worker]

    num_workers = 0
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
            if self.workers[worker].alive:
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

    def kill(self, index: int) -> None:
        if self.workers[index].alive:
            self.workers[index].parent_pipe.send((-1, -1))
            self.workers[index].parent_pipe.close()
            self.workers[index].process.terminate()
            self.workers[index].process.join()
            self.workers[index].needs_update = False
            self.num_workers -= 1
            self.workers[index].alive = False
            print(f"[{self.TaskName} MANAGER]: Killed a worker with {self.num_workers} remaining")


# each process needs about 250mb with standard settings
def conv_manager() -> WorkerManager:  # req. for full usage: 1.8 GByte with 0.3 GByte being reserve
    return WorkerManager("convolution search", [], .3, 6, 0, sh.worker_convsearch)


def span_manager() -> WorkerManager:  # req. for full usage: 0.6 GByte with 0.1 GByte being reserve
    return WorkerManager("span search", [], .3, 2, 1, sh.worker_spansearch)


def concat_manager() -> WorkerManager:
    return WorkerManager("concatenation search", ["convolution search", "span search"], .3, 6, 2, sh.worker_concatsearch)


########################################################################################################################
########################################################################################################################
########################################################################################################################


class ProcessHandler:
    # constants

    def __init__(self, loader_cfg: dict,
                 managers: List[WorkerManager],
                 samples: dict,
                 maxram: float = -1):

        self.loader_cfg = loader_cfg
        self.managers = self.order_tasks(managers)
        self.samples = samples
        self.sample_keys = list(samples.keys())
        self.maxram = maxram
        self.allocated_ram = 0

        self.working_managers = []
        self.fulfilled_tasks = []
        self.orders_and_searches = {}
        self.explanations = {}

    # static methods:
    @staticmethod
    def order_tasks(managers: List[WorkerManager]) -> List[WorkerManager]:  # simple bubble sort as task length is very small
        for _ in range(len(managers) - 1):
            for i in range(len(managers) - 1):
                if managers[i].TaskIndex > managers[i + 1].TaskIndex:
                    managers[i], managers[i + 1] = managers[i + 1], managers[i]

        for i in range(len(managers)):
            managers[i].TaskID = i

        return managers

    @staticmethod
    def iterator(listlike) -> any:
        for i in range(len(listlike)):
            yield listlike[i]
        while True:
            yield -1
    # end of static methods

    def get_available_ram(self) -> float:
        return min(psutil.virtual_memory().available / (1024**3), self.maxram if self.maxram >= 0 else sys.maxsize)

    def check_requirements(self, manager: WorkerManager) -> bool:
        req = True
        if manager.RequiredTasks:
            for i in manager.RequiredTasks:
                if i not in self.fulfilled_tasks:
                    req = False
        return req

    def __call__(self, *args, **kwargs) -> Tuple[dict, dict]:
        while sum([manager.done for manager in self.managers]) != len(self.managers):
            if self.allocated_ram > 0:
                print(f"[MAIN]: Freed {self.allocated_ram}GB of memory from previous searches")
            self.allocated_ram = 0
            for manager in self.managers:
                if manager.TaskName not in self.fulfilled_tasks:
                    print(f"[MAIN]: Starting [{manager.TaskName} MANAGER]")
                    self.start_manager(manager)
                    self.orders_and_searches[manager.TaskName] = {}
                    self.explanations[manager.TaskName] = {}

            working = True
            print(f"[MAIN]: Waiting for manager(s), allocated {self.allocated_ram}GB of RAM to "
                  f"{sum([manager.num_workers for manager in self.working_managers])} processes")

            while working:
                working = False
                for manager in self.working_managers:
                    self.check_manager(manager)
                    if manager.active:
                        working = True

            self.working_managers = []
        return self.orders_and_searches, self.explanations

    def generate_worker(self, manager: WorkerManager) -> Worker:
        parent, child = mp.Pipe()
        process = mp.Process(target=manager.Task, args=(self.loader_cfg["sgn"],
                                                        self.loader_cfg["len_filters"],
                                                        self.loader_cfg["metric"],
                                                        child),
                             daemon=True,
                             name=manager.TaskName)
        return Worker(process, parent, child, True)

    def check_manager(self, manager: WorkerManager) -> None:
        if manager.num_workers > 0:  # move this to another method
            result, workers_without_work = manager.get()
            for entry in result:
                if entry[0]:
                    self.orders_and_searches[manager.TaskName][entry[2]] = entry[0]
                if entry[1]:
                    self.explanations[manager.TaskName][entry[2]] = entry[1]
            for worker in workers_without_work:  # TODO: split into 2 methods
                if manager.active:
                    key, value = self.get_workerargs(manager)
                    manager.set(worker, (key, value))
                    if key == -1:
                        manager.kill(worker)
        else:
            manager.active = False
            manager.done = True
            self.fulfilled_tasks.append(manager.TaskName)

    def get_workerargs(self, manager: WorkerManager) -> tuple:
        if manager.TaskName == "span search":
            key = next(manager.iterator)
            return key, self.samples[key] if key != -1 else key

        if manager.TaskName == "convolution search":
            key = next(manager.iterator)
            return key, self.samples[key] if key != -1 else key

        if manager.TaskName == "concatenation search":
            key = next(manager.iterator)
            return key, self.samples[key] if key != -1 else key

        raise NotImplementedError  # TODO: modularize

    def start_manager(self, manager: WorkerManager) -> None:
        workers = []
        if self.check_requirements(manager):
            while len(workers) < manager.DesiredProcesses:
                available_ram = self.get_available_ram() - self.allocated_ram
                if available_ram > manager.RequiredRamPerProcess:
                    self.allocated_ram += manager.RequiredRamPerProcess
                    workers.append(self.generate_worker(manager))
                else:
                    break
            if len(workers) < 1:
                raise MemoryError(f"Not enough memory to start at least 1 process for [{manager.TaskName} MANAGER]")
            manager.workers = workers
            manager.num_workers = len(workers)
            manager.iterator = self.iterator(self.sample_keys)  # TODO: modularize
            manager.start()
            self.working_managers.append(manager)
        else:
            print(f"[MAIN]: Delaying start of [{manager.TaskName} MANAGER] due to unfulfilled requirements")

