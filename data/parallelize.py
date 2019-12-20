from multiprocessing.dummy import Pool
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent


def parallelize_with_results(func, data, workers=1):
    if workers is None or workers < 1:
        pool = Pool(processes=len(data))
    else:
        pool = Pool(processes=workers)
    results = []
    with tqdm(total=len(data)) as pbar:

        for i, res in tqdm(enumerate(pool.imap_unordered(func, data))):
            pbar.update()
            results.append(res)
    pool.close()
    pool.join()

    return results


def parallelize_without_results(func, datas, workers=1):
    process_pool = ThreadPoolExecutor(max_workers=workers)
    futures = {process_pool.submit(func, data): data for data in datas}

    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            print(result)
        except Exception as e:
            print('Something wrong happened in parallelize_without_results')
            print(e)
