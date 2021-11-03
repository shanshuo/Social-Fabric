# Social-Fabric
Social Fabric: Tubelet Compositions for Video Relation Detection

This repository contains the code and results for the following paper:  

**Social Fabric: Tubelet Compositions for Video Relation Detection** \
Shuo Chen, Zenglin Shi, Pascal Mettes and Cees G. M. Snoek \
ICCV 2021

## Dataset
Download VidOR validation annotation file `vidor_gt_val_relation.json` (<a href="https://surfdrive.surf.nl/files/index.php/s/0M5tuj5fPENkEmJ" target="_blank">link</a>) and put under `data/` folder.

## Evaluation
Download our method's result file on VidOR validation set `social_fabric_vidor_results.tar.gz` (<a href="https://surfdrive.surf.nl/files/index.php/s/im4cN3AABGSjD7B" target="_blank">link</a>) under `results/` folder and tar. Then run:
```
python evaluate/evaluator.py data/vidor_gt_val_relation.json results/social_fabric_vidor_results
```


If you find this code useful in your research please cite:
```
@inproceedings{chen2021social,
  title={Social Fabric: Tubelet Compositions for Video Relation Detection},
  author={Chen, Shuo and Shi, Zenglin and Pascal, Mettes and Snoek, Cees GM},
  booktitle={ICCV},
  year={2021}
}

