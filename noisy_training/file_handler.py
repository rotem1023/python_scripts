import os

def create_dir(dirpath, except_last =False):
    split_path = dirpath.split("/")
    cur_path = split_path[0]
    os.makedirs(cur_path, exist_ok=True)
    n = len(split_path)
    if except_last:
        n -=1
    for i in range(1,n):
        d = split_path[i]
        # if d == '.':
        #     continue
        cur_path = f"{cur_path}/{d}"
        os.makedirs(cur_path, exist_ok=True)
        

def get_data_dir(dataset):
    root_dir = '/home/dsi/rotemnizhar/dev/python_scripts/noisy_training'
    return f'{root_dir}/{dataset}'