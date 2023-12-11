
import os

# get today
import datetime
today = datetime.datetime.now() # only MMDD
today = today.strftime("%m%d")

# node_list = [
#     'ycheng-vpet',
#     'clin-test-perf',
#     'ycheng-eval',
#     'ycheng-vpet-n0',
#     'ycheng-vpet-n1',
#     'ycheng-vpet-n2',
#     'ycheng-vpet-n3',
#     'ycheng-vpet-n4',
# ]

node_list = ['ycheng-vpet-n1', 'ycheng-vpet-n2']

node_dict = { i: node_list[i] for i in range(len(node_list)) }


cmd_txt = f'train_gs_cmd_{today}.txt'
fp = open(cmd_txt, 'w')

N_obj = 1182

N_node = len(node_list)
N_gpus = 8

N_obj_per_gpu = N_obj // N_gpus // N_node

print("Total number of nodes: %d" % N_node)
print("Total number of GPUs: %d" % N_gpus)
print("Total number of objects: %d" % N_obj)
print("Number of objects per GPU: %d" % N_obj_per_gpu)

for i in range(N_node):

    print(" =============== Node %d =============== " % i)
    msg = f"# node {i}: {node_dict[i]}"
    fp.write(msg + '\n')

    for j in range(N_gpus):
        start_index = i * N_gpus * N_obj_per_gpu + j * N_obj_per_gpu
        end_index = start_index + N_obj_per_gpu
        cmd = f"python train_gs_batch.py --gpu_id {j} --start_index {start_index} --end_index {end_index}"
        # print(cmd)
        fp.write(cmd + '\n')

    # for i in range(N_gpus):
    #     start_index = i * N_obj_per_gpu
    #     end_index = (i + 1) * N_obj_per_gpu if i != N_gpus - 1 else N_obj
    #     cmd = f"./rend_objaverse.sh {i} {start_index} {end_index}"
    #     print(cmd)

fp.close()

print(cmd_txt)