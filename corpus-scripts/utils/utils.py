from multiprocessing.pool import ThreadPool
import time
import traceback
import requests


def run_threaded_for(func,iterable:list, args:list=None,log=False,threads:int=6):
    '''Runs a function for each value in iterable'''
    if not iterable:
        return

    # limit threads during working hours
    if log:
        start_time = time.time()
        print(f'Threaded: Running {func.__name__} to gather info from {len(iterable)} items | {threads} threads')

    iterable_divided = [None]*threads
    max_slice_size = round(len(iterable)/threads)
    # divide work between threads
    for i in range(threads):
        if i == threads-1:
            iterable_divided[i] = iterable[i*max_slice_size:]
        else:
            iterable_divided[i] = iterable[i*max_slice_size:(i+1)*max_slice_size]
            
    # if last threads has no work, remove it
    if not iterable_divided[-1]:
        iterable_divided = iterable_divided[:-1]
        threads -= 1
            

    thread_args = [[x]+args if args else [x] for x in iterable_divided]
    try:
        pool = ThreadPool(threads)
        results = pool.starmap(func,thread_args)
        pool.close()
        pool.join()
        if log:
            print(f'Threaded: Finished {func.__name__} in {time.time()-start_time} seconds')
    except Exception as e:
        print(f'Threaded: Error running {func.__name__}')
        print(traceback.format_exc())
        print(e)
        pool.terminate()
        raise e
    return results


def get_request(url,headers=None,params=None,retry:bool=True,sleep_time:int=1,retries:int=30):
    '''Make requests with auto retry'''
    ok_response = False
    response = None
    tries = 0
    while not ok_response and tries < retries:
        try:
            response = requests.get(url, headers=headers,params=params,timeout=30)
            if response.status_code == 200:
                ok_response = True
            # if too many requests, wait and retry
            elif response.status_code == 429 and retry:
                time.sleep(sleep_time)
            else:
                print(f'Error requesting {url}, status code: {response.status_code}, message: {response.text[:100]}')
                break
            tries += 1
        except Exception as e:
            print(f'Error requesting {url}, message: {str(e)[:100]}')
            tries += 1
            if retry and tries < retries:
                time.sleep(sleep_time)
            else:
                break
    if not ok_response and response:
        print(f'Error requesting {url}, status code: {response.status_code}, message: {response.text[:100]}')
    return response if ok_response else None


def clean_document_name(document_name):
    '''Clean document name'''
    replace_tokens = [
        '\n',
        ':',
        '\\',
        '/',
        '*',
        '?',
        '"',
        '<',
        '>',
        '|',
    ]
    document_name = document_name.strip()
    for token in replace_tokens:
        document_name = document_name.replace(token,'_')
    return document_name


def ifad_date_to_date(date):
    '''Convert ifad date (day[number] month[string] year[number]) to date'''
    months = {
        'jan' : '01',
        'feb' : '02',
        'mar' : '03',
        'apr' : '04',
        'may' : '05',
        'jun' : '06',
        'jul' : '07',
        'aug' : '08',
        'sep' : '09',
        'oct' : '10',
        'nov' : '11',
        'dec' : '12',
    }
    date = date.split(' ')
    data_month = date[1].lower()[:3]
    return f'{date[2]}-{months[data_month]}-{date[0]}'