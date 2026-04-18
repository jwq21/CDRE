import itertools
import random
from data import ImageDetectionsField, TextField, RawField
from data import DataLoader, Sydney, UCM, RSICD
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, RelationEnhanceAttention
import torch
from tqdm import tqdm
import argparse, os, pickle
import numpy as np
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

def predict_captions(model, dataloader, text_field):
    model.eval()
    gen = {}
    gts = {}
    image_names = {}
    
    device = next(model.parameters()).device
    
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(dataloader):
            detections = images[0].to(device)
            detections_gl = images[1].to(device)
            detections_mask = images[2].to(device)
            
            image_paths = dataloader.dataset.examples[it*dataloader.batch_size:(it+1)*dataloader.batch_size]
            
            with torch.no_grad():
                out, _ = model.beam_search(detections, detections_gl, detections_mask, 20,
                                           text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                sample_id = '%d_%d' % (it, i)
                gen[sample_id] = [gen_i, ]
                gts[sample_id] = gts_i
                
                if len(image_paths) > i:
                    image_name = os.path.basename(image_paths[i].image)
                    image_names[sample_id] = image_name
                
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores, gen, gts, image_names


if __name__ == '__main__':
    device = torch.device('cuda:3')

    parser = argparse.ArgumentParser(description='CDRE')
    parser.add_argument('--exp_name', type=str, default='Sydney')  # Sydney，UCM，RSICD
    # parser.add_argument('--exp_name', type=str, default='RSICD')  # Sydney，UCM，RSICD
    # parser.add_argument('--exp_name', type=str, default='UCM')  # Sydney，UCM，RSICD
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=0)
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


    args = parser.parse_args()

    print('Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(clip_features_path=args.clip_features_path,
                                       res_features_path=args.res_features_path,
                                       mask_features_path=args.mask_features_path)
    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)



    if args.exp_name == 'Sydney':
        dataset = Sydney(image_field, text_field, 'Sydney/images/', args.annotation_folder, split_file=args.split_file)
    elif args.exp_name == 'UCM':
        dataset = UCM(image_field, text_field, 'UCM/images/', args.annotation_folder, split_file=args.split_file)
    elif args.exp_name == 'RSICD':
        dataset = RSICD(image_field, text_field, 'RSICD/images/', args.annotation_folder, split_file=args.split_file)

    _, _, test_dataset = dataset.splits

    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))



    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=RelationEnhanceAttention)
    decoder = MeshedDecoder(len(text_field.vocab), 127, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)


    data = torch.load('/home/smhf/jwq/saved_models/new_huafen_USR_seed16/Sydney_best.pth')


    model.load_state_dict(data['state_dict'])
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores, gen, gts, image_names = predict_captions(model, dict_dataloader_test, text_field)
    print(f"评估分数: {scores}")


