import multiprocessing as mp
import multiprocessing.shared_memory as smm
import sys

import numpy as np
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
    Task: object  # preferably a method but anything callable works
    done = False
    associated_processes = []

    def __str__(self) -> str:
        return self.TaskName

    def __repr__(self) -> str:
        return f"Task {self.TaskName}"


def conv_task() -> TaskBase:
    return TaskBase("ConvSearch", [], 3, 4, 0, sm.convolution_search)


def span_task() -> TaskBase:
    return TaskBase("SpanSearch", [], 3, 1, 1, sm.span_search)


def concat_task() -> TaskBase:
    return TaskBase("ConcatSearch", ["ConvSearch", "SpanSearch"], 3, 2, 2, sm.concatenation_search)


########################################################################################################################
########################################################################################################################
########################################################################################################################


class ProcessHandler:
    #constants
    prime_delimiter = -2743

    def __init__(self, loader: Verbalizer,
                 tasks: List[TaskBase],
                 samples: dict):

        self.manager = mp.Manager()  # deprecated?
        self.root = loader
        self.tasks = self.order_tasks(tasks)
        self.samples = samples
        # we try reconstructing the dict after calculations
        self.sample_indices = smm.SharedMemory(create=True, size=sys.getsizeof(samples))
        self.sample_indices_buf = self.sample_indices.buf
        #  we buffer the attributions
        self.sample_attributions = smm.SharedMemory(create=True, size=sys.getsizeof(samples) + len(samples.keys()))
        self.sample_attributions_buf = self.sample_attributions.buf
        #  and we buffer the texts
        self.sample_texts = smm.SharedMemory(create=True, size=sys.getsizeof(samples) + len(samples.keys()))
        self.sample_texts_buf = self.sample_texts.buf
        # now instantiating the buffers  #
        self.sample_indices_buf[:] = list(samples.keys())[:]
        offset_indices_attributions = int(0)
        # fill buffer with attributions & use prime decompositional as delimiter
        for key in samples.keys():
            attrs = [*samples[key]["attributions"], self.prime_delimiter]
            self.sample_attributions_buf[:offset_indices_attributions + 1] = attrs[:]
            offset_indices_attributions += len(samples[key]["attributions"]) + 1

        # fill buffer with texts & use prime decompositional as delimiter
        offset_indices_texts = int(0)
        for key in samples.keys():
            attrs = [*samples[key]["input_ids"], self.prime_delimiter]
            self.sample_texts_buf[:offset_indices_texts + 1] = attrs[:]
            offset_indices_texts += len(samples[key]["attributions"]) + 1

        # continue working here 1
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
    def get_available_ram() -> float:
        return psutil.virtual_memory().available / (1024**3)

    @staticmethod
    def equalsplit_data(data: np.array, pieces: int) -> list:  # https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
        """Yield successive pieces-sized chunks from data."""
        for i in range(0, len(data), pieces):
            yield data[i:i + pieces]

    def attr_instance_generator(self) -> np.ndarray:
        item = None
        index = 0
        while index < len(self.sample_texts_buf):
            ret = []
            while self.sample_texts_buf[index] != self.prime_delimiter:
                ret.append(self.sample_texts_buf[index])
                index += 1
            if self.sample_texts_buf[index] == self.prime_delimiter:
                index += 1
            yield np.array(ret, dtype=np.float32)

    def text_instance_generator(self) -> List[int]:
        item = None
        index = 0
        while index < len(self.sample_texts_buf):
            ret = []
            while self.sample_texts_buf[index] != self.prime_delimiter:
                ret.append(self.sample_texts_buf[index])
                index += 1
            if self.sample_texts_buf[index] == self.prime_delimiter:
                index += 1
            yield ret

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

    def mainloop(self):
        pass

    def start_task(self, task: TaskBase) -> List[mp.Process]:
        processes = []
        if self.get_available_ram() > task.RequiredRamPerProcess:
            if self.check_requirements(task):
                num_procs = int(min(self.get_available_ram()/task.RequiredRamPerProcess, task.DesiredProcesses))

                for i in range(num_procs):
                    # continue working here 2
                    pass

        return processes





