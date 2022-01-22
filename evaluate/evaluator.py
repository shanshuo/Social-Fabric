import os
from collections import defaultdict
import numpy as np
import random
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from common import voc_ap, viou

def eval_detection_scores(gt_relations, pred_relations, viou_threshold, args):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    hit_scores = np.ones((len(pred_relations))) * -np.inf
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    for pred_idx, pred_relation in enumerate(pred_relations):
        ov_max = -float('Inf')
        k_max = -1
        this_ov_max = -float('Inf')
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx] \
                    and tuple(pred_relation['triplet']) == tuple(gt_relation['triplet']):
                s_iou = viou(pred_relation['sub_traj'], pred_relation['duration'],
                             gt_relation['sub_traj'], gt_relation['duration'])
                o_iou = viou(pred_relation['obj_traj'], pred_relation['duration'],
                             gt_relation['obj_traj'], gt_relation['duration'])
                ov = min(s_iou, o_iou)

                if ov > 0:
                    pred_relation['viou'] = ov
                    pred_relation['hit_gt_id'] = gt_idx

                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx

                if ov > this_ov_max:
                    this_ov_max = ov

        if k_max >= 0:
            hit_scores[pred_idx] = pred_relation['score']
            gt_detected[k_max] = True
            pred_relation['viou'] = ov_max
            pred_relation['hit_gt_id'] = k_max

    hit_scores = hit_scores[hit_scores != -1]
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def eval_tagging_scores(gt_relations, pred_relations):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    # ignore trajectories
    gt_triplets = set(tuple(r['triplet']) for r in gt_relations)
    pred_triplets = []
    hit_scores = []
    for r in pred_relations:
        triplet = tuple(r['triplet'])
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(r['score'])
    hit_scores = np.asarray(hit_scores, dtype=np.float32)
    for i, t in enumerate(pred_triplets):
        if not t in gt_triplets:
            hit_scores[i] = -np.inf
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def evaluate(gt_relations, prediction_root, viou_threshold, vid):
    predict_res_path = os.path.join(prediction_root, vid + '.json')
    try:
        with open(predict_res_path) as f:
            predict_relations = json.load(f)
            predict_relations = predict_relations[vid]
    except ValueError:
        print('load {} failed'.format(predict_res_path))
    det_prec, det_rec, det_scores = eval_detection_scores(
        gt_relations, predict_relations, viou_threshold, args)
    tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations)
    return vid, det_prec, det_rec, det_scores, tag_prec


def wrapper_evaluate(groundtruth, prediction, viou_threshold=0.5, det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """ evaluate visual relation detection and visual
    relation tagging.
    """
    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0

    print('Computing average precision AP over {} videos...'.format(len(groundtruth)))
    vid_num = len(groundtruth)

    results = Parallel(n_jobs=-1)(delayed(evaluate)(groundtruth[vid], prediction,
                                                    viou_threshold, vid)
                                 for vid in tqdm(groundtruth.keys()))

    for vid, det_prec, det_rec, det_scores, tag_prec in results:
        tot_gt_relations += len(groundtruth[vid])
        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        # compute precisions in tagging setting
        for nre in tag_nreturns:
            cut_off = min(nre, tag_prec.size)
            if cut_off > 0:
                prec_at_n[nre].append(tag_prec[cut_off - 1])
            else:
                prec_at_n[nre].append(0.)
    # calculate mean ap for detection
    # mean_ap = np.mean(list(video_ap.values()))
    mean_ap = round(float(np.mean(list(video_ap.values()))), 4)
    # calculate recall for detection
    rec_at_n = dict()
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_nreturns:
        mprec_at_n[nre] = np.mean(prec_at_n[nre])
    # print scores
    print('detection mean AP (used in challenge): {}'.format(mean_ap))
    print('detection recall@50: {}'.format(rec_at_n[50]))
    print('detection recall@100: {}'.format(rec_at_n[100]))
    print('tagging precision@1: {}'.format(mprec_at_n[1]))
    print('tagging precision@5: {}'.format(mprec_at_n[5]))
    print('tagging precision@10: {}'.format(mprec_at_n[10]))
    return mean_ap, rec_at_n, mprec_at_n


if __name__ == "__main__":
    """
    You can directly run this script from the parent directory, e.g.,
    python evaluation.visual_relation_detection val_relation_groundtruth.json val_relation_prediction_dir
    """
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Video visual relation detection evaluation.')
    parser.add_argument('groundtruth', type=str, help='A ground truth JSON file generated by yourself')
    parser.add_argument('prediction', type=str, help='A prediction files folder')
    # parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    # debug = args.debug
    # print(f'debug: {debug}')

    print('Loading ground truth from {}'.format(args.groundtruth))
    with open(args.groundtruth, 'r') as fp:
        gt = json.load(fp)
    print('Number of videos in ground truth: {}'.format(len(gt)))

    print('Loading prediction from {}'.format(args.prediction))
    print('Number of videos in prediction: {}'.format(len(os.listdir(args.prediction))))

    mean_ap, rec_at_n, mprec_at_n = wrapper_evaluate(gt, args.prediction)
