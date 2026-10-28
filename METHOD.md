# Method description

Team: GitRemoveAll   Track(s): 1 (this repository; Track 2 is in a separate repository)   Reported testing-phase submission(s): #19059 (user #2008)

## Approach (one page at most)

Two separate networks run on every image at the scoring resolution (longest side 1024, PIL bilinear, the evaluator's formula).
One produces the mask, the other the restored image. They share nothing, and each output depends only on its own input image.

Segmentation (`src/seg_model.py`) uses a frozen DINOv3 ViT-L/16 backbone and a small DPT-style head that we call DPT-lite. The head
takes the hidden states of blocks 6, 12, 18 and 24 (CLS and register tokens dropped), projects each to 256 channels, fuses them,
upsamples x2 twice and ends with a 1x1 classifier for the 5 classes. Only the head is trained. At test time we average the softmax
over scales 0.75 and 1.0, each with and without a horizontal flip, and take the argmax without any class bias. Portrait images are
rotated to landscape before the network and rotated back afterwards.

Restoration (`src/restore_big.py`, RestoreNet) is built on an ImageNet ConvNeXt-Tiny encoder from timm. The encoder sees the image
and, as 3 extra channels, a per-image min-max stretched copy of it. The network predicts a field of 3x4 colour-affine transforms at
H/8, upsamples it bilinearly and applies it to the input; full-resolution refinement blocks then add back detail. RestoreNet also
has a small head that estimates the Jerlov water type (10 classes) and the degradation level, and it can be conditioned on either
its own estimate or on given values. In `run.py` the type and level are read from the end of the file name
(`<scene>_<type>_<level>.png`); this is the "Meta" flag we set on the leaderboard. If a name does not end in `_<type>_<level>`, the
network uses its own estimate (`cond=None`) and nothing else changes.

The restored image is clipped to [0, 1], converted to 8 bit with rounding and then rounded to even values (`quantize`, step 2). We
had to do this last step to get the leaderboard zip under the 1024 MB upload limit. The rounded image is about 51 dB PSNR away
from the unrounded one. Masks are written as single-channel PNG with labels 0 to 4.

## Training data

EURS: the 20 labelled training scenes, all 200 degradations of each. The two submitted checkpoints are the "all" versions trained
on the 20 scenes. Iteration counts and hyper-parameters were chosen beforehand on a 16/4 scene split (validation scenes `fish13`,
`fish20`, `fish9`, `starfish2`). For restoration the ground truth was re-downsampled from the full-resolution training GT with
`cv2.INTER_AREA`. Development-phase and testing-phase images were not used for training, pseudo-labelling or any adaptation.

Segmentation was trained for 5,250 iterations with AdamW (head lr 3e-4), batch 4 with 2 accumulation steps, 576x768 crops and fp16.
The loss is CE + Dice, with uniform class weights in the last 20% of the iterations, plus an edge loss (weight 0.3). Augmentation:
rare-class crop sampling, physical re-degradation of clean images (p 0.2), vertical flip (p 0.3) and colour jitter.

External segmentation data, 30,619 images, labels remapped to the 5 EURS classes, with categories EURS does not define set to ignore:

| Dataset | Images | Licence |
|---|---|---|
| SUIM | 1,598 | MIT |
| USIS10K | 5,215 | Apache-2.0 |
| UIIS10K | 10,036 | Apache-2.0 |
| USIS16K | 11,695 | research use only |
| Coralscapes | 2,075 | Apache-2.0 |

An external image replaces a EURS sample with probability 0.5; images that contain elasmobranchs or "other" are drawn 4x more
often, and half of the external images are re-degraded with physical water parameters. External data was used for segmentation
only.

Restoration was trained for 12,000 iterations (lr 5e-4, encoder 5e-5, fp16) with progressive crops of 256x256, 384x512 and finally
576x768. Loss: PSNR (0.1) + cosine (0.5) + outlier-aware L1 (1.0) + 1 - SSIM (0.5) + L1 on the H/8 output (0.5); type and level
estimation add CE (0.1) and L1 (1.0). Blackout images keep weight 1.0. Augmentation is synthetic re-degradation of EURS GT (p 0.3).
No external images were used for restoration.

Pretrained models:
- `facebook/dinov3-vitl16-pretrain-lvd1689m` (Meta, DINOv3 License). Frozen; its weights are stored inside our segmentation checkpoint.
- timm `convnext_tiny`, ImageNet-1k (Apache-2.0). Initialises the RestoreNet encoder, fine-tuned on EURS.

## Anything non-standard at inference

Segmentation TTA (2 scales x horizontal flip, mean of the 4 softmax maps), file-name metadata for the restoration network, and the
even-value rounding of the restored images described above. No ensemble and no other post-processing.

Disclosure: #19059 is the best of our five testing-phase Track 1 submissions, so it was picked using leaderboard scores. Two of the
others used a different segmentation model and a colour post-process. The last two (28 October) tried a global saturation change on
the restored images and a constant seg-logit bias; both scored lower and neither is in the reported submission.

## Reproduction notes

- `requirements.txt` adds `transformers==4.57.6`. Transformers 5.x needs PyTorch >= 2.6; 4.57.6 runs DINOv3 on the base image's 2.4.0.
- The segmentation checkpoint was saved with transformers 5.x, where the blocks are named `bb.model.layer.N`. Version 4.57 calls them
  `bb.layer.N`, so `run.py` renames the prefix while loading. The tensors are the same.
- The DINOv3 backbone is built offline from the bundled `config.json` (`AutoModel.from_config`). All its weights are in
  `seg_dinov3_vitl16_ext_all.pt`, so nothing is downloaded at run time.
- The weights are one zip behind the link in `weights/weights_link.yaml` (SHA-256
  `bf9a8764b525e9760c80a031f8c02f830a5da68862404d13cc7d3b74ef72b668`). It unpacks into `weights/` as
  `seg_dinov3_vitl16_ext_all.pt`, `dinov3_vitl16_config/config.json`, `restore_big_all.pt`, `LICENSE_DINOv3.md` (the DINOv3 licence
  has to travel with redistributed DINOv3 weights) and `THIRD_PARTY_LICENSES.txt`.
- The leaderboard run was on a Kaggle Tesla T4: segmentation under fp16 autocast, restoration in fp32. On another GPU a few mask
  pixels near class boundaries and the last bit of a few restored values may change.
- We checked `run.py` against the #19059 zip on a Kaggle T4. Restored images are byte-identical on all 1,200 images. Masks are
  pixel-identical on a 40-image check (agreement 1.000000), and stay identical with transformers 5.x and `from_pretrained`.
- Run time on a T4 is about 2.1 s per image, so about 42 min for the 1,200 testing-phase images, using roughly 6 GB of GPU memory.
  Images are decoded one at a time; some inputs are tens of megapixels.
- Training code is not needed for reproduction; it is available on request.
