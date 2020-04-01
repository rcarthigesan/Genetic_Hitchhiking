"""
Created on 09/02/2019 12:55

@author: R Carthigesan

"""

import numpy as np
import os
import datetime

test_name = "dl_prog"

prog_text = open(test_name + ".txt", "r")
n_sites = 5e5


def tail(f, lines=1, _buffer=4098):
    """Tail a file and get X lines from the end"""
    # place holder for the lines found
    lines_found = []

    # block counter will be multiplied by buffer
    # to get the block size from the end
    block_counter = -1

    # loop until we find X lines
    while len(lines_found) < lines:
        try:
            f.seek(block_counter * _buffer, os.SEEK_END)
        except IOError:  # either file is too small, or too many lines requested
            f.seek(0)
            lines_found = f.readlines()
            break

        lines_found = f.readlines()

        block_counter -= 1

    return lines_found[-lines:]


lastline = tail(prog_text)[0]
prog_text.close()
n_tasks = int(((lastline.rsplit())[-5]))
time_taken = ((lastline.rsplit())[-1])

percent_remaining = np.round((100 * n_tasks/n_sites), 1)
time_remaining = np.round((float(time_taken[:-3]) * ((100 - percent_remaining) / percent_remaining)), 1)

hours = int(time_remaining // 60)
minutes = int((time_remaining - hours * 60))

print(str(percent_remaining) + "% complete")
print("Time taken: " + time_taken)
print("Estimated time remaining: " + str(hours) + "h " + str(minutes) + "m")
print("Estimated finish time: " + str(datetime.datetime.now() + datetime.timedelta(minutes = time_remaining))[:-7])