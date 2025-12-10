# HAKSNet: Hierarchical Adaptive Kernel-Spatio Network for Remote Sensing Visual Tasks

## Introduction
This repository contains the official implementation of our HAKSNet, a backbone network tailored for remote sensing visual tasks.  
HAKSNet introduces hierarchical adaptive kernel selection, spatial selection, and redundancy suppression, effectively capturing multi-scale spatial textures and semantic representations in RSIs.

## Results and Models

### Pretrained Models
- ImageNet pretrained HAKSNet-T backbone: **[Download](#)**  
- ImageNet pretrained HAKSNet-S backbone: **[Download](#)**  

---

## Experiments Results

### DOTA-v1.0
| Model | mAP | Angle | Aug | Configs | Download |
|-------|------|--------|--------|----------|-----------|
| HAKSNet-T (1024,1024,200) | **XX.XX** | le90 | - | [haks-t_fpn_dotav1_le90](configs/haks/haks-t_fpn_dotav1_le90.py) | [model](#) |
| HAKSNet-S (1024,1024,200) | **XX.XX** | le90 | - | [haks-s_fpn_dotav1_le90](configs/haks/haks-s_fpn_dotav1_le90.py) | [model](#) |

---

### DOTA-v1.5
| Model | mAP | Angle | Aug | Configs | Download |
|-------|------|--------|--------|----------|-----------|
| HAKSNet-S (1024,1024,200) | **XX.XX** | le90 | - | [haks-s_fpn_dotav15_le90](configs/haks/haks-s_fpn_dotav15_le90.py) | [model](#) |

---

### HRSC2016
| Model | mAP | Angle | Aug | Configs | Download |
|-------|------|--------|--------|----------|-----------|
| HAKSNet-S | **XX.XX** | le90 | - | [haks-s_fpn_hrsc_le90](configs/haks/haks-s_fpn_hrsc_le90.py) | [model](#) |

---

### FAIR1M
| Model | mAP | Angle | Aug | Configs | Download |
|-------|------|--------|--------|----------|-----------|
| HAKSNet-T | **XX.XX** | le90 | - | [haks-t_fpn_fair_le90](configs/haks/haks-t_fpn_fair_le90.py) | [model](#) |

---

## Citation
If you find HAKSNet helpful, please cite our work:

```bibtex
@article{your2025haksnet,
  title={HAKSNet: Hierarchical Adaptive Kernel-Spatio Network for Remote Sensing Visual Tasks},
  author={Your Name},
  journal={Journal / Conference},
  year={2025}
}
