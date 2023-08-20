import multiprocessing as mpr
import threading as thr
import time
from typing import List, Optional


class Timer:
    def __init__(self, time_list: List[float]):
        self.times = time_list
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self.times.append(time.time() - self.start_time)


class TicToc:
    __t_start = {}

    def __init__(self, title="", debug=True):
        self.title = title
        self.should_print = debug

    def tic(self, _print=False):
        mp_name = self.mp_name
        times = TicToc.__t_start.setdefault(mp_name, [])

        if _print and self.should_print:
            indent_level = len(times)
            indent_str = self._get_indent_str(indent_level)
            print(f"{indent_str}{mp_name}::[{self.title}]", flush=True)
        times.append(time.time())

    def toc(self):
        mp_name = self.mp_name
        times = TicToc.__t_start[mp_name]

        t_elapsed = time.time() - times.pop()
        indent_level = len(times)
        indent_str = self._get_indent_str(indent_level)
        if self.should_print:
            print(f"{indent_str}{mp_name}::[{self.title}] complete in {t_elapsed:.3f}s", flush=True)

    def toc_val(self):
        mp_name = self.mp_name
        times = TicToc.__t_start.setdefault(mp_name, [])
        return time.time() - times.pop()

    @property
    def mp_name(self):
        return f"{mpr.current_process().name}.{thr.current_thread().name}"

    @staticmethod
    def _get_indent_str(level):
        return "--" * level

    def __enter__(self):
        self.tic(_print=True)

    def __exit__(self, type, value, traceback):
        self.toc()
