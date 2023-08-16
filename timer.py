import time

class TimerError(Exception):
    """timer exception"""


class Timer():

    def __init__(self):
        self._start_time = None

    def __enter__(self):
        if self._start_time is not None :  
            raise TimerError("you need to use .stop() before starting")

        self._start_time = time.pref_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        
        