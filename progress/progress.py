import time as tm
import sys
import numpy as np
_tt=0
_ss=''

intervals = (
    ('w', 604800),  # 60 * 60 * 24 * 7
    ('d', 86400),    # 60 * 60 * 24
    ('h', 3600),    # 60 * 60
    ('m', 60),
    ('s', 1),
    )


def display_time(seconds, granularity=2):
    result = []
    if seconds <= 0:
        return '0s'
    for name, count in intervals:
        value = seconds // count
        if (name == 's') and not result:
            value = round(seconds, max(int(-np.log10(seconds)),0)+1 )
            result.append("{}{}".format(value, name))
        elif value:
            seconds -= value * count
            value = int(value)
            result.append("{}{}".format(value, name))
    if not result:
        return '0s'
    return ', '.join(result[:granularity])


def start(s="", end="\n", beg="\r", add = ''):
    global _tt
    global _ss
    _ss = s
    if s != "":
        print(beg + s + " ... " + add, end=end)
    _tt= tm.time()
    return _tt

def tick(s='', i=-1, n=0, end="\n", beg="\r", eta=False, add = ''):
    s_eta = ''
    if s == '':
        s = _ss
    if eta:
        if i==0:
            s_eta = ''
        else:
            s_eta = ' (ETA: ' + display_time( (tm.time() -_tt) / i * (n-i) ) +')' 
    if i<0:
        print(beg + s + ' ' +add + "               " , end=end)
    else:
        print(beg + s + (" " if s !="" else "") + str(i+1) + ((" out of " + str(n)) if n!=0 else "") + s_eta + ' ' + add + "          ", end=end)


def finish(s="", t=0, beg="\r", add = ''):
    if s == '':
        s = _ss
    if t == 0:
        t = _tt
    print(beg + s + (" - " if s !="" else "") + "finished in: " + display_time(tm.time() - t) + ' ' + add + "          ")