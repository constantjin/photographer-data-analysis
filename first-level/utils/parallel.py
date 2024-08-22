import multiprocessing

import parmap

def pmap(function, iterable, *args, **kwargs):
    try:
        pool = multiprocessing.Pool()
        return parmap.map(function, iterable, *args, **kwargs, pm_pool=pool)
    finally:
        pool.terminate()
        pool.join()