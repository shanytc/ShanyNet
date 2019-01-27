import sys

sys.path.append('/ib/junk/junk/shany_ds/shany_proj')

from app import app
from shani_main import infer_folder, Cfg, ShanyNet, BasicBlock
import os

def inferImage(folder=None):
    if folder is None:
        return []

    cfg = Cfg()
    cfg.test_data_dirpath = '/Users/i337936/Documents/shany_net/shany_net/dataset/test' if _DEV_ == True else \
        '/ib/junk/junk/shany_ds/shany_proj/dataset/test'
    cfg.train_data_dirpath = '/Users/i337936/Documents/shany_net/shany_net/dataset/train' if _DEV_ == True else \
        '/ib/junk/junk/shany_ds/shany_proj/dataset/train'
    cfg.train_save_model_path = '/Users/i337936/Documents/shany_net/shany_net/model/model.h5' if _DEV_ == True else \
        '/ib/junk/junk/shany_ds/shany_proj/model/model.h5'

    cfg.train_net_cfg[1]['num_classes'] = len(os.listdir(cfg.train_data_dirpath))

    # inference
    cfg.test_model_path = cfg.train_save_model_path
    cfg.infer_folder_classes_list = sorted(os.listdir(cfg.train_data_dirpath))
    cfg.test_data_dirpath = '/ib/junk/junk/shany_ds/shany_proj/dataset/inference/'+folder
    res = infer_folder(cfg)
    return res

_DEV_ = False

if __name__ == '__main__':


    #res = inferImage('test/')
    #print(res)
    app.debug = True
    app.run(host = '0.0.0.0', port=5005, debug=False)
