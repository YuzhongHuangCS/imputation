import os
import concurrent.futures
import pdb
import sys

executor = concurrent.futures.ThreadPoolExecutor(48)
filename = sys.argv[1]
save_prefix = sys.argv[2]

for i in range(48):
	cmd = f'taskset -c {i} python3 -u {filename}.py {i} {save_prefix} > log/{save_prefix}_{i}.log'
	print(cmd)
	executor.submit(os.system, cmd)

print('All task generated')
executor.shutdown()
print('All done')
