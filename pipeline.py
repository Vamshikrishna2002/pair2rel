
def pipeline(text,entity_labels,rel_labels):
    import json
    from tqdm import tqdm
    import pickle as pkl
    import argparse
    # import networkx as nx
    # python -m spacy download en_core_web_sm

    from gliner import GLiNER
    import os
    import torch
    model=GLiNER.from_pretrained("urchade/gliner_large-v2.1")
    # Force usage of GPU 1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    import json
    if entity_labels is None:
        with open('entities.json', 'r') as f:
            real_entities = json.load(f)
        entity_label=real_entities['entity']

    entities=model.predict_entities(text,labels=entity_label,threshold=0.3,multi_label=True)
    print('entity extraction done')
    print(entities)
    ner=[]
    import re
    dtokens=re.split(r'[ ,.]+', text)
    for entity in entities:
        ent_text=re.split(r'[ ,]+', entity['text'])
        start=dtokens.index(ent_text[0])
        end=dtokens.index(ent_text[-1])
        ner.append([start, end, entity['text'],entity['label']])

    
    from pair2rel.model import Pair2Rel

    rel_model = Pair2Rel.from_pretrained("chapalavamshi022/pair2rel", cache_dir="/tmp/hf_cache",force_download=True)

    import torch

    # Force usage of GPU 1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    rel_model = rel_model.to(device)
    rel_model.device = device 
    rel_labels = rel_labels['relation_labels']
    relations = rel_model.predict_relations(dtokens, rel_labels, threshold=0.1,ner=ner, top_k=1,flat_ner=True)

    fine_relations = []
    for i in range(len(ner)-1):
        for rel in relations:
            # print(rel,ner[i],ner[i+1])
            head=' '.join(rel['head_text'])
            tail=' '.join(rel['tail_text'])
            if head==ner[i][2] and tail==ner[i+1][2]:
                fine_relations.append((ner[i][2], rel['label'], ner[i+1][2]))
    print(fine_relations)
            
    print("Success! ✅")

