import random

from torch.backends import cudnn

from data import ImageDetectionsField, TextField, RawField
from data import DataLoader, Sydney, UCM, RSICD
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, RelationEnhanceAttention
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile

import warnings

warnings.filterwarnings("ignore")



def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_torch()


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections, detections_gl, detections_mask, captions) in enumerate(dataloader):
                detections, detections_gl, detections_mask, captions = detections.to(device), detections_gl.to(device), \
                                                                       detections_mask.to(device), captions.to(device)
                out = model(detections, detections_gl, detections_mask, captions, isencoder=True)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
    val_loss = running_loss / len(dataloader)
    return val_loss


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            detections = images[0].to(device)
            detections_gl = images[1].to(device)
            detections_mask = images[2].to(device)

            with torch.no_grad():
                out, _ = model.beam_search(detections, detections_gl, detections_mask, 20,
                                           text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, detections_gl, detections_mask, captions) in enumerate(
                dataloader):

            detections, detections_gl, detections_mask, captions = detections.to(device), detections_gl.to(device), \
                                                                   detections_mask.to(device), captions.to(device)
            out = model(detections, detections_gl, detections_mask, captions, isencoder=True)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
            loss.backward()

            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    return loss


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            detections = images[0].to(device)
            detections_gl = images[1].to(device)
            detections_mask = images[2].to(device)

            outs, log_probs = model.beam_search(detections, detections_gl, detections_mask, 20,
                                                text_field.vocab.stoi['<eos>'], beam_size, out_size=beam_size)

            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline



if __name__ == '__main__':
    device = torch.device('cuda:3')
    parser = argparse.ArgumentParser(description='CDRE')
    parser.add_argument('--exp_name', type=str, default='Sydney')  # Sydney，UCM，RSICD
    # parser.add_argument('--exp_name', type=str, default='UCM')  # Sydney，UCM，RSICD
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)  # 1000
    parser.add_argument('--warm_up_epochs', type=int, default=10)  # 10
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--split_file', type=str, default='/home/smhf/jwq/testCDRE/New_CDRE/CDRE/pre_CLIP/dataset_splits_IMAGE_BASED/Sydney_split_indices_seed16_IMAGE_BASED.json', help='Path to the JSON file with pre-split dataset indices.')


    ################################################################################################################

    # Sydney
    parser.add_argument('--annotation_folder', type=str,
                        default='/home/smhf/jwq/testCDRE/New_CDRE/CDRE/features/Sydney')
    parser.add_argument('--res_features_path', type=str,
                        default='/home/smhf/jwq/CDRE/dataset/multi_scale_T/sydney_res152_con5_224')

    parser.add_argument('--clip_features_path', type=str,
                        default='/home/smhf/jwq/testCDRE/New_CDRE/CDRE/dataset/clip_feature/sydney_224_seed16_fair_huafenSydney')

    parser.add_argument('--mask_features_path', type=str,
                        default='/home/smhf/jwq/CDRE/dataset/multi_scale_T/sydney_group_mask_8_6')

    # UCM
    # parser.add_argument('--annotation_folder', type=str,
    #                     default='/home/smhf/jwq/testCDRE/New_CDRE/CDRE/features/UCM')
    # parser.add_argument('--res_features_path', type=str,
    #                     default='/home/smhf/jwq/CDRE/dataset/multi_scale_T/UCM_res152_con5_224')

    # parser.add_argument('--clip_features_path', type=str,
    #                     default='/home/smhf/jwq/CDRE/dataset/clip_feature/UCM_224_huafenUCM')

    # parser.add_argument('--mask_features_path', type=str,
    #                     default='/home/smhf/jwq/CDRE/dataset/multi_scale_T/UCM_group_mask_8_6')
    
    # RSICD
    # parser.add_argument('--annotation_folder', type=str,
    #                     default='/home/smhf/jwq/testCDRE/New_CDRE/CDRE/features/RSICD')
    # parser.add_argument('--res_features_path', type=str,
    #                     default='/home/smhf/jwq/CDRE/dataset/multi_scale_T/RSICD_res152_con5_224')

    # parser.add_argument('--clip_features_path', type=str,
    #                     default='/home/smhf/jwq/CDRE/dataset/clip_feature/RSICD_224')

    # parser.add_argument('--mask_features_path', type=str,
    #                     default='/home/smhf/jwq/CDRE/dataset/multi_scale_T/RSICD_group_mask_8_6')






    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    args = parser.parse_args()
    print(args)

    print('Transformer Training')
    # log
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field = ImageDetectionsField(clip_features_path=args.clip_features_path,
                                       res_features_path=args.res_features_path,
                                       mask_features_path=args.mask_features_path)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)



    # train_dataset, val_dataset, test_dataset = dataset.splits
    if args.exp_name == 'Sydney':
        dataset = Sydney(image_field, text_field, 'Sydney/images/', args.annotation_folder, split_file=args.split_file)
    elif args.exp_name == 'UCM':
        dataset = UCM(image_field, text_field, 'UCM/images/', args.annotation_folder, split_file=args.split_file)
    elif args.exp_name == 'RSICD':
        dataset = RSICD(image_field, text_field, 'RSICD/images/', args.annotation_folder, split_file=args.split_file)


    train_dataset, val_dataset, test_dataset = dataset.splits



    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))


    encoder = MemoryAugmentedEncoder(3, 0, attention_module=RelationEnhanceAttention)
    decoder = MeshedDecoder(len(text_field.vocab), 127, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)
        # return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -2)



    # Initial conditions
    optim = Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    # best_test_cider = .0
    patience = 0
    start_epoch = 0

    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/Top_K_17_seed16/%s_last.pth' % args.exp_name
        else:
            fname = 'saved_models/Top_K_17_seed16/%s_best.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']

    print("Training starts")
    
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                       num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)
        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train,
                                                             text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)


        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu2', scores['BLEU'][1], e)
        writer.add_scalar('data/val_bleu3', scores['BLEU'][2], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)
        writer.add_scalar('data/val_spice', scores['SPICE'], e)
        writer.add_scalar('data/val_S*',
                          (scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr']) / 4, e)
        writer.add_scalar('data/val_Sm', (
                scores['BLEU'][3] + scores['METEOR'] + scores['ROUGE'] + scores['CIDEr'] + scores['SPICE']) / 5, e)


        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                # print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True


        if switch_to_rl and not best:
            data = torch.load('saved_models/Top_K_17_seed16/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])


        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,

            'use_rl': use_rl,
        }, 'saved_models/Top_K_17_seed16/%s_last.pth' % args.exp_name) # save last model ；saved_models 是保存模型的目录。；%s_last.pth 是保存的文件名，其中 %s 会被 args.exp_name 替换。

        if best:
            copyfile('saved_models/Top_K_17_seed16/%s_last.pth' % args.exp_name, 'saved_models/Top_K_17_seed16/%s_best.pth' % args.exp_name)

        if exit_train:
            writer.close()
            break

