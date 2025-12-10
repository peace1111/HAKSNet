# HAKSNet: Hierarchical Adaptive Kernel-Spatio Network for Remote Sensing Object Detection

## Abstract
Recent backbone networks tailored for remote sensing images mainly rely on larger receptive fields or multi-scale convolutions to enhance feature extraction. However, they overlook the layer-wise variation in importance of different kernel sizes, limiting their adaptability to fine-grained textures in shallow layers and complex semantics in deeper layers. We propose HAKSNet that dynamically selects optimal kernel sizes and spatial attention at each layer. A dedicated redundancy suppression mechanism using partial convolution and gating further reduces background noise typical in RSIs. Extensive experiments on four challenging benchmarks across object detection, semantic segmentation, and change detection demonstrate HAKSNet’s superior performance and strong generalization. 

## Results and Models

### Pretrained Models
- ImageNet pretrained **HAKSNet-T** backbone:  
  👉 [Download](https://pan.baidu.com/s/1el7v8DrMlwtrHBQAqDt-oA?pwd=qc4r)  （提取码: qc4r）

- ImageNet pretrained **HAKSNet-S** backbone:  
  👉 [Download](https://pan.baidu.com/s/1jRcvoAj3AfKXJkeduzuQnw?pwd=gq7n)  （提取码: gq7n)  

---

## Experiments Results

### DOTA-v1.0-SS
| Model | mAP | Angle | Aug | Configs | Download |
|-------|------|--------|--------|----------|-----------|
| HAKSNet-T (1024,1024,200) | **79.28** | le90 | - | [haks-t_fpn_dotav1_le90ss](https://pan.baidu.com/s/1f5wcFRqVbRM16bBmlpRH7w?pwd=9nfa)（提取码: 9nfa) | [model](https://pan.baidu.com/s/1KCITs2hSGPnC1rKaC0njPw?pwd=rvch)（提取码: rvch) |
| HAKSNet-S (1024,1024,200) | **80.20** | le90 | - | [haks-s_fpn_dotav1_le90ss](configs/haks/haks-s_fpn_dotav1_le90.py) | [model](#) |

### DOTA-v1.0-MS
| Model | mAP | Angle | Aug | Configs | Download |
|-------|------|--------|--------|----------|-----------|
| HAKSNet-T (1024,1024,200) | **81.16** | le90 | - | [haks-t_fpn_dotav1_le90ms](configs/haks/haks-t_fpn_dotav1_le90.py) | [model](#) |
| HAKSNet-S (1024,1024,200) | **81.86** | le90 | - | [haks-s_fpn_dotav1_le90ms](configs/haks/haks-s_fpn_dotav1_le90.py) | [model](#) |

---

