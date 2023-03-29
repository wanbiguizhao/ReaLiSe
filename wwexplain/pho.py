# 研究语音模型
import os 
import sys 
PROJECT_DIR=os.path.dirname(
    os.path.dirname(
        os.path.realpath(__file__)
    )
)
sys.path.append(PROJECT_DIR)
from transformers import BertConfig,BertTokenizer
from src import Pho2ResPretrain,Pho2Pretrain,create_dataset,data_helper,make_features
import torch
class Args:
    def __init__(self) -> None:
        pass
def data_work(args):
    train_dataset = create_dataset(args, args.train_file)
    t_total = len(train_dataset) // args.train_batch_size // args.gradient_accumulation_steps * args.num_train_epochs
    for step, batch in enumerate(data_helper(args, train_dataset, tokenizer, batch_processor, False)):
        pass

def instance_pho():
    args=Args()
    args.data_dir="data"
    args.model_name_or_path="pretrained/"
    args.train_file="trainall.times2.pkl"
    args.no_cuda=False
    args.train_batch_size=2
    args.gradient_accumulation_steps=1
    args.num_train_epochs=20
    args.max_seq_length=512
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    config_class, model_class, tokenizer_class = BertConfig,Pho2Pretrain,BertTokenizer
    config = BertConfig.from_pretrained("pretrained/config.json")
    tokenizer=BertTokenizer("pretrained/vocab.txt")
    model=Pho2Pretrain.from_pretrained("pretrained")
    batch_processor = Pho2Pretrain.build_batch

    model.to(args.device)


    train_dataset = create_dataset(args, args.train_file)
    t_total = len(train_dataset) // args.train_batch_size // args.gradient_accumulation_steps * args.num_train_epochs
    for step, batch in enumerate(data_helper(args, train_dataset[:96], tokenizer, batch_processor, False)):
        for t in batch:
            if t not in ['id', 'src', 'tgt', 'lengths', 'tokens_size', 'pho_lens']:
                batch[t] = batch[t].to(args.device)
        loss = model(batch)[0]

    
if __name__=='__main__':
    #print(args.data_dir)
    instance_pho() 
