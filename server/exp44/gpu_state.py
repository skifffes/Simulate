import subprocess

print(subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                      '--format=csv,noheader'], capture_output=True, text=True).stdout)
print(subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                      '--format=csv,noheader'], capture_output=True, text=True).stdout)
print(subprocess.run(['df', '-h', '/scratch'], capture_output=True, text=True).stdout)
print(subprocess.run(['bash', '-lc', 'whoami; id -u'], capture_output=True, text=True).stdout)
