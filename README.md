# HAKSNet: Hierarchical Adaptive Kernel-Spatio Network for Remote Sensing Object Detection

## Abstract
Recent backbone networks tailored for remote sensing images mainly rely on larger receptive fields or multi-scale convolutions to enhance feature extraction. However, they overlook the layer-wise variation in importance of different kernel sizes, limiting their adaptability to fine-grained textures in shallow layers and complex semantics in deeper layers. We propose HAKSNet that dynamically selects optimal kernel sizes and spatial attention at each layer. A dedicated redundancy suppression mechanism using partial convolution and gating further reduces background noise typical in RSIs. Extensive experiments on four challenging benchmarks across object detection, semantic segmentation, and change detection demonstrate HAKSNet’s superior performance and strong generalization. 

## Results and Models

### Pretrained Models
- ImageNet pretrained **HAKSNet-T** backbone:  
  👉 [Download](https://github.com/)  

- ImageNet pretrained **HAKSNet-S** backbone:  
  👉 [Download](https://github.com/)  

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
