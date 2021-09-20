import os
import json

from config import settings


def log(args, ACC_list, name, sup=''):
    save_log_dir = os.path.join(settings.SNAPSHOT_DIR, args.arch, args.seed)
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)
    filename = os.path.join(save_log_dir, name + sup + '.json')
    '''
    log_file = open(os.path.join(save_log_dir, name+sup+'.txt'), 'w')
    log_file.write(str(ACC_list))
    log_file.close()
    '''
    with open(filename, 'w') as file_obj:
        json.dump(ACC_list, file_obj)


