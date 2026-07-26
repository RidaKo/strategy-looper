

from multiprocessing import Pool
import struct
import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

class mdata:
    tick_size = None
    big_point_value = None
    country = None
    exchange = None
    symbol = None
    description = None
    interval_type = None
    interval_span = None
    time_zone = None
    session = None
    data: pd.DataFrame = None

    def __repr__(self):
        return str((self.symbol, self.interval_type, self.interval_span, self.data.shape))

def read_ts_ohlcv_dat_one(fname) -> mdata:
    try:
        f = open(fname, "rb")
        d = read_ts_ohlcv_dat_one_stream(f)
    finally:
        f.close()
    return d

def read_ts_ohlcv_dat_one_stream(byte_stream) -> mdata:
    def read_string(f):
        sz = struct.unpack('i', f.read(4))[0]
        s = f.read(sz).decode('ascii')
        return s
    d = mdata()
    try:
        # f = open(fname, "rb")
        f = byte_stream
        (ones, type_format) = struct.unpack('ii', f.read(8))
        if (ones != 1111111111):
            print("format not supported, must be 1111111111")
            return None
        if (type_format != 3):
            print("type_format not supported, must be 3")
            return None
        d.tick_size = struct.unpack('d', f.read(8))[0]
        d.big_point_value = struct.unpack('d', f.read(8))[0]
        d.country = read_string(f)
        d.exchange = read_string(f)
        d.symbol = read_string(f)
        d.description = read_string(f)
        d.interval_type = read_string(f)
        d.interval_span = struct.unpack('i', f.read(4))[0]
        d.time_zone = read_string(f)
        d.session = read_string(f)

        dt = np.dtype([('date', 'f8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('volume', 'f8')])
        #data = np.fromfile(f, dtype=dt)
        z = f.read()
        #data = np.frombuffer(z, dtype=dt)
        #data = pd.DataFrame.from_records(data)
        data = pd.DataFrame.from_records(np.frombuffer(z, dtype=dt))
    finally:
        #f.close() #no need to close - -will be closed outside
        pass

    arr = ((data['date']-25569)*24*60*60).round(0).astype(np.int64)*1000000000
    #25569 - difference between datenum('2088-02-25') - datenum('2018-02-23')
    # .round(0) - to remove millisecond error
    # *1000000000 - to convert to native type of nanoseconds
    z2 = pd.to_datetime(arr)
    data.insert(0, 'datetime', z2)
    del data['date']
    d.data = data
    return d

def read_ts_ohlcv_dat(fnames) -> List[mdata]:
    r = []
    for name in glob.glob(fnames):
        print('loading ', name)
        z = read_ts_ohlcv_dat_one(name)
        r.append(z)
    print('done')
    return r

class spnl:
    data = None
    fname = None
    symbol = None
    dir = None
    strat = None
    session = None
    trade_count = None
    bar_size = None
    param = None
    symbol_root = None
    sidb = None
    date = None
    pnl = None


########################################################################################################################
def read_ts_pnl_dat_one(fname)->spnl:
    """

    :param fname: string with file name or bytes with data
    :return:
    """

    if type(fname) is bytes:
        f = fname
        (ones, type_format) = struct.unpack('ii', f[:8])
        dt = np.dtype([('date', 'f4'), ('pnl', 'f4'), ('pos', 'f4'), ('trade', 'f4')])
        data = np.frombuffer(f[8:], dtype=dt)
        additional_info = ['a','a','a','a','a','a']
        sida = 'a_a_a_a_a_a'
    elif type(fname) is str:
        try:
            f = open(fname, "rb")
            (ones, type_format) = struct.unpack('ii', f.read(8))
            dt = np.dtype([('date', 'f4'), ('pnl', 'f4'), ('pos', 'f4'), ('trade', 'f4')])
            data = np.fromfile(f, dtype=dt)
            #data = pd.DataFrame.from_records(data)
            additional_info = os.path.basename(fname).split('_')
            sida = fname.split('\\')[-1].replace('.dat','')
        finally:
            f.close()
    else:
        f = fname
        fname = os.path.basename(f.name)
        (ones, type_format) = struct.unpack('ii', f.read(8))
        dt = np.dtype([('date', 'f4'), ('pnl', 'f4'), ('pos', 'f4'), ('trade', 'f4')])
        data = np.frombuffer(f.read(), dtype=dt)
        additional_info = os.path.basename(fname).split('_')
        sida = fname.split('\\')[-1].replace('.dat','')



    #arr = np.floor(data['date'] - 25569).astype(np.int64) * 24 * 60 * 60 * 1000000000
    #z2 = pd.to_datetime(arr)
    #data.insert(0, 'datetime', z2)
    #del data['date']


    d = spnl()

    #d.date = pd.to_datetime(np.floor(data['date'] - 25569).astype(np.int64) * 24 * 60 * 60 * 1000000000).values
    #data['date'] =
    d.date = np.array(np.floor(data['date'] - 25569).astype(np.int64) * 24 * 60 * 60 * 1000000000, dtype='datetime64[ns]')
    d.pnl = data['pnl']

    #d.data = data
    #d.date = data.datetime
    #d.pnl = data.pnl
    d.fname = fname
    d.symbol = additional_info[0]
    d.dir = additional_info[1]
    d.strat = additional_info[2]
    d.session = additional_info[3]
    d.trade_count = data['trade'].max()
    d.bar_size = additional_info[4]
    d.param = additional_info[5]
    d.symbol_root = additional_info[0]
    d.sida = sida
    k = d.sida.split('_')
    d.sidb = '_'.join(k[0:1] + k[2:-1])
    d.pos = data['pos']
    d.trade = data['trade']

    return d
########################################################################################################################
def read_ts_pnl_dat(fnames):
    print('loading ', fnames)
    r = []
    counter = 0
    for name in glob.glob(fnames):
        #print('loading ', name)
        try:
            z = read_ts_pnl_dat_one(name)
            r.append(z)
            # print('done')
        except:
            print("cannot read :", name)
        else:
            counter += 1
    print(f"loaded files: {counter}")
    return r
########################################################################################################################
def read_ts_pnl_dat_multicore(fnames):
    print(f'load started from {fnames}', end =" ")

    import threading
    import time

    files = glob.glob(fnames)

    print(f', files: {len(files)}', end =" ")

    time_start = time.time()

    def read_one_file(one_file, results, index):
        results[index] = read_ts_pnl_dat_one(one_file)

    thread_list = [None] * len(files)
    results = [None] * len(files)
    for i in range(len(files)):
        thread_list[i] = threading.Thread(target=read_one_file, args=(files[i], results, i))
        thread_list[i].start()

    for o in thread_list:
        o.join()

    print(f"done. elapsed: {(time.time() - time_start):7} seconds")
    return results
###############################################################
def read_ts_pnl_dat_multicore_old(fnames, number_of_threads=16):
    """
    plase use this form: if __name__ == '__main__':
    :param fnames:
    :return:
    """
    print('load started from ' + fnames)
    files = glob.glob(fnames)
    ilgis = len(files)
    chunksize = ilgis // number_of_threads
    agents = number_of_threads
    pool = Pool(agents)
    r = pool.map(read_ts_pnl_dat_one, files, chunksize)
    pool.close()
    print('done')
    return r
###############################################################
def read_ts_pnl_csv_one(fname: str) -> pd.DataFrame:
    a = pd.read_csv(fname, delimiter=',', index_col=0, parse_dates=True, dayfirst=False, header=None,
                    names=['pnl', 'c', 'd', 'e'])
    a.index = np.array(a.index, dtype='datetime64[D]')
    a.index.name = 'date'
    a.fname = fname
    a.sida = fname.split('\\')[-1].replace('.CSV', '')
    k = a.sida.split('_')
    a.sidb = '_'.join(k[0:1] + k[2:-1])
    a.symbol = k[0]
    return a
########################################################################################################################
def read_ts_pnl_csv(fnames: "") -> List[pd.DataFrame]:
    r = []
    for name in glob.glob(fnames):
        print('loading ', name)
        z = read_ts_pnl_csv_one(name)
        r.append(z)
    print('done')
    return r
########################################################################################################################
def allign_numpy(d) -> Tuple[np.ndarray, np.ndarray]:
    all_date = np.unique(np.concatenate([t.date for t in d]))
    all_date = np.array(all_date, dtype='datetime64[D]')
    all_pnl_matrix = np.zeros((len(d), len(all_date)), dtype='float32')
    i = 0
    for q in d:
        idx = np.in1d(all_date, q.date, assume_unique=True)
        all_pnl_matrix[i, idx] = q.pnl
        i = i + 1

    return all_date, all_pnl_matrix
########################################################################################################################
def allign_pandas(d) -> Tuple[np.ndarray, pd.DataFrame]:
    all_date = np.unique(np.concatenate([t.date for t in d]))
    all_date = np.array(all_date, dtype='datetime64[D]')
    df1 = pd.DataFrame(0.0, index=all_date, columns=[a.sidb for a in d])
    for q in d:
        df1.iloc[np.in1d(all_date, q.date, assume_unique=True), df1.columns.get_loc(q.sidb)] = q.pnl

    return all_date, df1
########################################################################################################################
def mavg_old_and_slow(d,p):
    r = np.zeros(len(d))
    r[p - 1] = d[:p].sum()
    for i in range(p, len(d)):
        r[i] = r[i - 1] - d[i - p] + d[i]
    return r / p
########################################################################################################################
def mavg(d, period):
    return np.concatenate([np.zeros((period-1)), np.convolve(d, np.ones((period,))/period, mode='valid')])
########################################################################################################################
def emavg(d : np.ndarray,period: int) -> np.ndarray:
    if period == 1:
        return d
    r = np.zeros(len(d))
    alpha = 2 / (period + 1)  # weight for ema
    # first exponential average is a first price
    r[0] = d[1]
    for j in range(1, len(d)):
        r[j] = r[j - 1] + alpha * (d[j] - r[j - 1])
    return r
########################################################################################################################
def smooth(data_ : np.ndarray, period : int) -> np.ndarray:
    #r = np.zeros(len(data_))
    r = data = data_
    #data: np.ndarray = data_
    for i in range(0,period):
        r[0] = data[0]
        r[-1] = data[-1]
        r[1:-1] = (data[:-2] + data[1:-1] + data[2:])/3
        data = r
    return data
########################################################################################################################
def remove_bad_is(data, oos_date):

    if len(data) < 10000:
        print(f"too few data: {len(data)} - will not delete")
        return data

    bad_res = []
    for i in range(len(data)):
        a1 = data[i]
        idx_oos_date = np.argmax(a1.date >= oos_date)
        if a1.pnl[:idx_oos_date].sum() <= 0:
            bad_res.append(i)

    print(f"bad ones: {len(bad_res)} will be deleted")

    del_percent = len(bad_res) / len(data) * 100
    if del_percent > 90:
        print(f"too many deletes: {del_percent}% - will not delete")
        return data

    #delete
    import os
    for i in bad_res:
        try:
            os.remove(data[i].fname)
        except OSError:
            pass

    #remove from list
    a3 = np.array(data)
    a4 = np.delete(a3, bad_res)
    return a4
########################################################################################################################
