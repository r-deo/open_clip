import open_clip
import logging
import torch
import os
import random
import numpy as np
import argparse
import torch.nn.functional as F
from open_clip_train.inference_tool import (zeroshot_evaluation,
                            retrieval_evaluation,
                            semantic_localization_evaluation,
                            get_preprocess,
                            zeroshot_get_dataset,
                            # zeroshot_classifier
)
from open_clip_train.inference import build_model, evaluate


class EvalAll:
    def __init__(self, ckpt_path, model="ViT-B-32", device='cuda', test_data_dir='/home/ubuntu/geof/ITRA/itra/GeoRSCLIP/data/rs5m_test_data',
                 zero=1, pretrained='openai',batch_size=64 ,workers=8):
        self.model, self.img_preprocess = build_model(model, ckpt_path, device, zero)
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        self.workers = workers
        self.device = device


    def evaluate(self):
        # eval_result = evaluate(self.model, self.img_preprocess)
        print("making val dataset with transformation: ")
        print(self.img_preprocess)
        zeroshot_datasets = [
            'EuroSAT',
            'RESISC45',
            'AID'
        ]
        selo_datasets = [
            'AIR-SLT'
        ]

        self.model.eval()
        all_metrics = {}

        # zeroshot classification
        metrics = {}
        for zeroshot_dataset in zeroshot_datasets:
            zeroshot_metrics = self.zeroshot_evaluation(zeroshot_dataset)
            metrics.update(zeroshot_metrics)
            all_metrics.update(zeroshot_metrics)
        return all_metrics
    
    def zeroshot_evaluation(self, zeroshot_dataset):

        dataset = zeroshot_get_dataset(dataset_name=zeroshot_dataset, split='test', root=self.test_data_dir, transform=self.img_preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers)

        logging.info(f'Calculating classifier for {zeroshot_dataset}')
        classnames, prompt_templates = dataset.classes, dataset.templates
        import copy
        classnames = copy.deepcopy(classnames)
        classifier = self.zeroshot_classifier(classnames, prompt_templates)

        logging.info(f'Calculating image features for {zeroshot_dataset}')
        results = {}
        acc, features, labels = self.zeroshot_run(classifier, dataloader)
        logging.info(f'{zeroshot_dataset} zero-shot accuracy: {acc}%')
        results[f'{zeroshot_dataset}-zeroshot-acc'] = acc

        for key, item in results.items():
            results[key] = float(item)

        return results


    def zeroshot_accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        return float(correct[0].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) * 100 / len(target)


    def zeroshot_run(self, classifier, dataloader):
        import tqdm
        with torch.no_grad():
            all_image_features = []
            all_labels = []
            all_logits = []
            for images, target in tqdm.tqdm(dataloader, unit_scale=self.batch_size):
                images = images.to(self.device)
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1).detach().cpu()
                logits = 100. * image_features @ classifier
                all_image_features.append(image_features)
                all_labels.append(target)
                all_logits.append(logits)

        all_image_features = torch.cat(all_image_features)
        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)

        acc = self.zeroshot_accuracy(all_logits, all_labels, topk=(1,))
        return round(acc, 2), all_image_features, all_labels

    def zeroshot_classifier(self, classnames, templates):
        tokenizer = open_clip.tokenize
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [template.replace('{}', classname) for template in templates]
                context_length = 77
                texts = tokenizer(texts, context_length=context_length).to(self.device)

                class_embeddings = self.model.encode_text(texts)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embedding = F.normalize(class_embeddings, dim=-1)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding.cpu())
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        return zeroshot_weights


if __name__ == '__main__':
    ea = EvalAll("/data/logs/2024_11_29-18_06_54-model_ViT-B-32-lr_1e-06-b_64-j_8-p_amp/checkpoints/epoch_2.pt")
    print(ea.evaluate())