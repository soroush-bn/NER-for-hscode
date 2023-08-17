import time

class TimerError(Exception):
    """timer exception"""


class Timer():

    def __init__(self,class_name):
        self._start_time = None
        self.class_under_estimation =class_name 

    def __enter__(self):
        if self._start_time is not None :  
            raise TimerError("you need to use .stop() before starting")

        self._start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time of {self.class_under_estimation}: {elapsed_time:0.4f} seconds")
        
        