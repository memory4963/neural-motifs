import matplotlib
import cv2
import json
import numpy as np
from config import BOX_SCALE
from config import ModelConfig
conf = ModelConfig()


def __get_objs(pred_entry, labels, k=10):
    arr = np.array(pred_entry['obj_scores'])
    idxs = arr.argsort()[-k:][::-1]
    names = [labels['idx_to_label'][str(pred_entry['pred_classes'][x])] for x in idxs]
    boxes = [pred_entry['pred_boxes'][x] for x in idxs]
    scores = [arr[x] for x in idxs]
    objs = {}
    for idx, ori_idx in enumerate(idxs):
        objs[str(ori_idx)] = {
            'name': names[idx],
            'box': boxes[idx],
            'score': scores[idx]
        }
    return objs

def __get_rels(pred_entry, labels, objs, k=20):
    obj_idxs = [int(x) for x in list(objs.keys())]
    rel_idxs = []
    # 和物体有关的关系的idx
    for idx, rel in enumerate(pred_entry['pred_rel_inds']):
        if rel[0] in obj_idxs and rel[1] in obj_idxs:
            rel_idxs.append(idx)
    # 确定关系的类别和分数
    names = []
    scores = []
    for idx in rel_idxs:
        logits = np.array(pred_entry['rel_scores'][idx])
        # 除 above 外的其他类别（above的权重过高）
        rel_clz = logits.argsort()[-2:][::-1][1] + 1
        names.append(labels['idx_to_predicate'][str(rel_clz)])
        scores.append(logits[rel_clz - 1])
    scores = np.array(scores)
    # 此list中为rel_idxs的idx
    rel_idxs_idxs = [x for x in scores.argsort()[-k:][::-1]]
    names = [names[x] for x in rel_idxs_idxs]
    scores = [scores[x] for x in rel_idxs_idxs]
    fin_rel_idxs = [rel_idxs[x] for x in rel_idxs_idxs]

    rels = {}
    for idx, ori_idx in enumerate(fin_rel_idxs):
        in_obj = pred_entry['pred_rel_inds'][ori_idx][0]
        out_obj = pred_entry['pred_rel_inds'][ori_idx][1]
        rels[str(ori_idx)] = {
            'name': names[idx],
            'score': scores[idx],
            'in_obj': in_obj,
            'out_obj': out_obj,
            'svo': '{}{} {} {}{}'.format(in_obj, objs[str(in_obj)]['name'], names[idx], out_obj, objs[str(out_obj)]['name'])
        }
    return rels


def __generate_graph(pred_entry, obj_k=10, rel_k=20):
    '''
    graph: {
        objs: {
            idx: {
                name, box, score
                }},
        rels: [
            {name, score, in_obj, out_obj, svo: "1obj 11verb 4obj"}
        ]}
    svo: subject verb object
    '''
    with open('data/stanford_filtered/VG-SGG-dicts.json', 'r') as f:
        # keys: ['object_count', 'idx_to_label', 'predicate_to_idx', 'predicate_count', 'idx_to_predicate', 'label_to_idx']
        labels = json.load(f)

    objs = __get_objs(pred_entry, labels, obj_k)
    rels = __get_rels(pred_entry, labels, objs, rel_k)
    return {
        'objs': objs,
        'rels': rels
    }


def __draw_boxes(img, graph):
    if conf.custom_data:
        scale = 1
    else:
        scale = max(img.shape) / BOX_SCALE
    for item in graph['objs'].items():
        key, obj = item
        box = obj['box']
        cv2.rectangle(img, (int(box[0]*scale), int(box[1]*scale)), (int(box[2]*scale), int(box[3]*scale)), (0, 0, 255), 2)
        cv2.putText(img, '{}{}'.format(key, obj['name']), (int(box[0]*scale), int(box[1]*scale)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imwrite('figure1.jpg', img)


def visualize(fpath, pred_entry):
    img = cv2.imread(fpath)
    graph = __generate_graph(pred_entry)
    __draw_boxes(img, graph)

    for rel in graph['rels'].values():
        if rel['name'] == 'above':
            continue
        print(rel['svo'])


if __name__ == '__main__':
    with open('test.txt', 'r') as f:
        pred_entry = json.load(f)
    visualize('data/test/scene_eg2.jpg', pred_entry)
