import json
import requests
import numpy as np
import time
import threading


def single_post(data, url, timeout=5):
    
    data = json.dumps({"data":data,
                        "extra_info":['10']*6})
    
    # test time elapse
    t0 = time.time()
    response = requests.post(url, data=data, timeout=timeout)
    if response.status_code == 200:
        result = json.loads(response.text)['result']
    else:
        print(f'Error code: {response.status_code}')
        print(json.loads(response.text))
        return
    t1 = time.time()

    result = np.asarray(result)
    if result.shape[1:] == (540,960):
        print('single post test passed!!!')
        total = t1-t0
        print(f'execution time: {total}s')
        print()
    else:
        print('single post test failed')

    
def multi_post(data, url, num_thread=6, timeout=5):
    threads = []
    for i in range(num_thread):
        threads.append(threading.Thread(target=single_post, args=(data, url, timeout)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
if __name__ =="__main__":
    url = 'http://127.0.0.1:5000/steel_seg'
    single_post_data = [
        "/storage/unload_ratio/new_data/full_images/皖KV3551_20220607/皖KV3551_20220607124459.jpg",
    ]*6
    single_post(data=single_post_data, url=url, timeout=10)
    multi_post(data=single_post_data, url=url, timeout=20)
    