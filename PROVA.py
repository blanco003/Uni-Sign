
from config import *
from datasets import S2T_Dataset_news
import argparse
import utils as utils
from pathlib import Path
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader


def main(args):
    


    # No distribuited
    #utils.init_distributed_mode_ds(args)

    #print(args)
    utils.set_seed(args.seed)

    print(f"CREATING DATASET :")


    train_data = S2T_Dataset_news(path=train_label_paths[args.dataset], args=args, phase='train')
    print("\nTrain : ", train_data)

    """
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    """
    ##################################
    train_sampler = RandomSampler(train_data)
    ##################################
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=True)

 
    dev_data = S2T_Dataset_news(path=dev_label_paths[args.dataset],args=args, phase='dev')
    print("\nTest : ", dev_data)

    """
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    """
    #####################################
    dev_sampler = SequentialSampler(dev_data)
    #####################################
    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn,
                                 sampler=dev_sampler, 
                                 pin_memory=args.pin_mem)


    print("\nESEMPIO GET ITEM : ")
    
    sample_idx = 10
    sample = train_data[sample_idx]

    # Estrai i campi restituiti
    name_sample, pose_sample, text, gloss, support_rgb = sample

    print(f"\n[ESEMPIO SAMPLE #{sample_idx}]")
    print(f"Nome video: {name_sample}")
    print(f"Testo: {text}")
    print(f"Glossa: {gloss}")
    print(f"Keypoints: {list(pose_sample.keys())}")
    print(f"Pose shape : {pose_sample['body'].shape}")
    
    if support_rgb:
        print(f"Support RGB keys: {list(support_rgb.keys())}")
        print(f"Left hand shape: {support_rgb['left_hands'].shape}")


    print(f"CREATING MODEL :")

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n----------------------------Avvio--------------------------------")
    main(args)





# PER AVVIARE DA CONDA : 

# D:
# cd Uni-Sign-Main
# conda activate Uni-Sign
# script\PROVA.bat

# script\train_stage1.bat
# script\test_stage1.bat


# pip install decord==0.6.0
# pip install numpy==1.22.4