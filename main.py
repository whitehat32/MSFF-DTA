# -*- encoding: utf-8 -*-


import argparse
import os
import shutil

import pytorch_lightning as pl
import torch
import pandas as pd
import random
import core
from core import DeepCPIModel
from torch_geometric.data import Data, Batch
import os
# from lightning.pytorch.strategies import DDPStrategy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run(args: argparse.Namespace):
    """进行模型训练（测试）

    Args:
        args: 程序运行所需的命令行参数
    """
    model = DeepCPIModel(args)
    if args.objective == 'classification':
        monitor, mode = 'auc', 'max'
    else:
        monitor, mode = 'mse', 'min'
    #early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(min_delta=0,
    #                                                                patience=args.early_stop_round,
    #                                                                verbose=True,
    #                                                                monitor=monitor,
    #                                                                mode=mode)

    checkpoints_path = args.ckpt_save_path
    #checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=checkpoints_path,
    #                                                                    save_top_k=1,
    #                                                                    verbose=True,
    #                                                                    monitor=monitor,
    #                                                                    mode=mode)
    global os
    if args.mode == 'train':
        if not os.path.exists(checkpoints_path):
            os.mkdir(checkpoints_path)
        #else:
        #    shutil.rmtree(checkpoints_path)
        #    os.mkdir(checkpoints_path)
            
        trainer = pl.Trainer(
            default_root_dir=checkpoints_path,
            max_epochs=args.max_epochs,
            # strategy=DDPStrategy(find_unused_parameters=True),
            strategy='ddp_find_unused_parameters_true',
            devices=args.gpus,
            accelerator='gpu',
        )
        trainer.fit(model, ckpt_path=args.ckpt_path)
        trainer.save_checkpoint(os.path.join(checkpoints_path, 'final.ckpt'))
        trainer.test(model, ckpt_path=os.path.join(checkpoints_path, 'final.ckpt'))
        # trainer.test(model, ckpt_path=os.path.join(checkpoints_path, checkpoint_callback.best_model_path))

    elif args.mode == 'predict':
        trainer = pl.Trainer(
            default_root_dir='../',
            accelerator='gpu',
            devices=args.gpus
        )
        # if args.valid_test:
        #     valid_result = trainer.test(model, ckpt_path=args.ckpt_path)

        # trainer.test(model, ckpt_path=args.ckpt_path)
        dct = dict()
        def mpredict(mode):
            predictions = trainer.predict(model,  ckpt_path=args.ckpt_path)
            prediction_scores = []
            for p in predictions:
                print(p.shape)
                predictions = p.numpy()
                # import pdb; pdb.set_trace()
                prediction_scores.append(predictions)
            # print(predictions.shape)
            import numpy as np
            dct['prediction'] = np.concatenate(prediction_scores, axis=0)
            # _, indeices = torch.topk(predictions, k=30, dim=0, largest=True)
            # print(predictions[indeices])
            # print(indeices)
            # dct['indeices'] = indeices + 1
            
            smile_name = pd.read_csv('case_study_true.csv')
            dct['COMPOUND_SMILES'] = smile_name['COMPOUND_SMILES']

            print(len(dct)) 
            ans = pd.DataFrame(dct)
            ans.to_csv('indeices.csv', index=None)
            # predictions, _ = torch.topk(predictions, k=1, dim=0, largest=True)
            # print(predictions)
        import os

        # 获取当前工作目录
        current_path = os.getcwd()

        print("当前工作目录是:", current_path)
        mpredict(model)
        



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='Running Mode (train / test / predict)')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default=None,
                        help='CheckPoint File Path for Test')
    parser.add_argument('--ckpt_save_path', type=str, default='ckpts/debug', help='CheckPoint File Path for save')

    # parser.add_argument('--root_data_path', type=str, default='data/Kiba/drug_cold_start', help='Raw Data Path')
    parser.add_argument('--root_data_path', type=str, default='./', help='Raw Data Path')
    parser.add_argument('--objective',
                        type=str,
                        default='regression',
                        help='Objective (classification / regression)')

    parser.add_argument('--seed', type=int, default=2021, help='Random Seed')

    parser.add_argument('--gpus', type=str, default='1', help='Number of GPUs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for Train(Validation/Test)')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max Trainning Epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of Subprocesses for Data Loading')
    parser.add_argument('--learning_rate', type=float, default=4e-6, help='Learning Rate for Trainning')
    # parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning Rate for Trainning')
    parser.add_argument('--early_stop_round', type=int, default=5, help='Early Stopping Round in Validation')

    parser.add_argument('--decoder_layers', type=int, default=3, help='Number of Layers for Decoder')
    parser.add_argument('--n_heads', type=int, default=3, help='Number of Heads for Attention')
    parser.add_argument('--gnn_layers', type=int, default=3, help='Layers of GNN')
    parser.add_argument('--protein_gnn_dim', type=int, default=64, help='Hidden Dimension for Attention')
    parser.add_argument('--compound_gnn_dim', type=int, default=34, help='Hidden Dimension for Attention')
    parser.add_argument('--mol2vec_embedding_dim', type=int, default=300, help='Dimension for Mol2vec Embedding')
    parser.add_argument('--hid_dim', type=int, default=64, help='Hidden Dimension for Attention')
    parser.add_argument('--pf_dim', type=int, default=256, help='Hidden Dimension for Positional Feed Forward')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout Rate')

    parser.add_argument('--protein_encoder_layers', type=int, default=3, help='protein encoder Layers')
    parser.add_argument('--protein_encoder_head', type=int, default=4, help='protein encoder head')
    parser.add_argument('--cnn_kernel_size', type=int, default=7, help='CNN Conv Layers')
    parser.add_argument('--valid_test', type=bool, default=False, help='Testing for validation data')

    parser.add_argument('--protein_dim', type=int, default=64, help='Dimension for Protein')
    parser.add_argument('--atom_dim', type=int, default=34, help='Dimension for Atom')
    parser.add_argument('--edge_dim', type=int, default=6, help='Dimension for edge')
    parser.add_argument('--protein_embedding_dim', type=int, default=1280, help='Dimension for Protein Embedding')
    parser.add_argument('--compound_embedding_dim', type=int, default=2727, help='Dimension for drug Embedding')
    return parser.parse_args()


def check_data_set():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    tr_mp = dict()
    te_mp = dict()
    for i in range(len(train_data)):
        tr_mp[train_data['PROTEIN_NAME'][i]] = tr_mp.get(train_data['PROTEIN_NAME'][i], 0) + 1
    for i in range(len(test_data)):
        te_mp[test_data['PROTEIN_NAME'][i]] = te_mp.get(test_data['PROTEIN_NAME'][i], 0) + 1


    print('(key,value) pairs in common:', tr_mp.items() & te_mp.items())


if __name__ == '__main__':
    #调试
    params = parse_args()
    print(params)
    pl.seed_everything(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    run(params)





