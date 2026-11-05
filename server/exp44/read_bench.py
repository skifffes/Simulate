import json

for f in ('/scratch/lijh/exp42/curve2400/training_speed_benchmark.json',
          '/scratch/lijh/exp42/curve1800/training_speed_benchmark.json',
          '/scratch/lijh/exp42/ab_B/training_speed_benchmark.json'):
    j = json.load(open(f))
    print(f)
    for k in ('steps', 'optimizer_updates', 'batch_size', 'seconds_per_optimizer_step',
              'seconds_per_microbatch', 'compile_seconds'):
        if k in j:
            print('  ', k, j[k])
    print('  keys', list(j))
