# Method description

Team: GitRemoveAll   Track(s): 1 (this repository; Track 2 is in a separate repository)   Reported testing-phase submission(s): #19059 (user #2008)

## Approach (one page at most)

Two independent networks process every image at the scoring resolution, with the longest side resized to 1024 using PIL bilinear.

**Segmentation** (`src/seg_model.py`) is a frozen **DINOv3 ViT-L/16** backbone with a light DPT-style head (DPT-lite).
- The head reads the hidden states of blocks 6, 12, 18 and 24, dropping the CLS and register tokens.
- Each level is projected to 256 channels; the levels are then fused and upsampled x2 twice, followed by a 1x1 classifier for the 5 classes.
- Only the head is trained.
- Inference uses test-time augmentation: scales 0.75 and 1.0, each with and without a horizontal flip, and the four softmax maps are averaged.
- Portrait images are rotated to landscape for the network and rotated back.
- The mask is the argmax of the averaged softmax, with no class bias.

**Restoration** (`src/restore_big.py`, *RestoreNet*) works in three parts:
- A timm **ConvNeXt-Tiny** encoder, ImageNet-pretrained, extracts features. Its input is the image plus a per-image min-max stretched copy as 3 extra channels.
- The network predicts a spatial field of 3x4 colour-affine transforms at H/8, upsamples it, and applies it to the input.
- Full-resolution refinement blocks then sharpen the result.

The network is conditioned on the Jerlov **water type** (10 classes) and **degradation level**:
- It contains an estimation head for both, trained with the labels encoded in the training file names.
- At inference, type and level are read from the test file name `<scene>_<type>_<level>.png`; this is declared as "Meta" on the leaderboard.
- If a name does not contain them, `run.py` falls back to the network's own estimate (`cond=None`), with no other change.

**Output:**
- The restored image is clipped to [0, 1], converted to 8 bit with rounding, and then rounded to even values (`quantize`, step 2). This rounding was applied to the leaderboard zip to stay under the 1024 MB upload limit (about 51 dB vs the un-rounded image).
- Masks are saved as single-channel PNG (labels 0–4).
- Each output depends only on its own input image.

## Training data

- **EURS:** the 20 labelled training scenes, all 200 degradations of each.
  - Both submitted models are the *all* versions, trained on the 20 scenes.
  - Their iteration counts and hyper-parameters come from runs on a 16/4 scene fold: train on 16, validate on `fish13, fish20, fish9, starfish2`.
  - For restoration, the GT was re-downsampled from the original full-resolution GT images of the training set with `cv2.INTER_AREA`.
  - The development-phase and testing-phase images were **not** used for training, pseudo-labelling or any adaptation.
- **Segmentation**, 5,250 iterations:
  - Optimisation: AdamW, head lr 3e-4, batch 4 x 2 accumulation, crops of 576x768, fp16.
  - Losses: CE + Dice, with uniform class weights in the last 20% of iterations, plus an edge loss (weight 0.3).
  - Augmentation: minority-class sampling, synthetic physical re-degradation (p 0.2) and vertical flip (p 0.3).
  - External segmentation data (sampled with probability 0.5, re-degraded with physical water parameters): 30,619 images, with labels remapped to the 5 EURS classes and ambiguous categories set to ignore.
    - SUIM (MIT)
    - USIS10K (Apache-2.0)
    - UIIS10K (Apache-2.0)
    - USIS16K (research use)
    - Coralscapes (Apache-2.0)
- **Restoration**, 12,000 iterations:
  - Optimisation: lr 5e-4, encoder lr 5e-5, fp16.
  - Progressive crops: 256x256, then 384x512, then 576x768.
  - Losses: PSNR (0.1), cosine (0.5), outlier-aware L1 (1.0), 1 − SSIM (0.5), and an auxiliary L1 on the H/8 output (0.5). Type/level estimation uses CE (0.1) and L1 (1.0). Blackout images are weighted 1.0.
  - Augmentation: synthetic re-degradation of EURS GT (p 0.3). No external images.
- **Pretrained models:**
  - `facebook/dinov3-vitl16-pretrain-lvd1689m`, under the DINOv3 License (Meta). It is frozen, and its weights are included in our checkpoint.
  - timm `convnext_tiny` (ImageNet-1k), Apache License 2.0; RestoreNet encoder initialisation, fine-tuned on EURS.

## Anything non-standard at inference

- Segmentation test-time augmentation: 2 scales x horizontal flip, mean of the four softmax maps.
- Filename metadata (water type and level) conditions the restoration network.
- Restored images are rounded to even values to match the uploaded zip.
- No ensembling across models, and no post-processing beyond the above.

**Disclosure:** #19059 was the best of our three testing-phase Track 1 submissions, so it was selected on leaderboard feedback. The other two differed in the segmentation model and in a colour post-process.

## Reproduction notes

- `requirements.txt` adds `transformers==4.57.6`. Transformers 5.x needs PyTorch ≥ 2.6, while 4.57.6 runs DINOv3 on the base image's 2.4.0.
- The segmentation checkpoint was saved with transformers 5.x, which names the blocks `bb.model.layer.N`; in 4.57 they are `bb.layer.N`. `run.py` renames this prefix when loading, and the tensors are unchanged.
- The DINOv3 backbone is built offline from the bundled `config.json` (`AutoModel.from_config`). All its weights are in `seg_dinov3_vitl16_ext_all.pt`, so no network access is needed.
- Weights are behind the link in `weights/weights_link.yaml`, as a single zip that unpacks into `weights/`:
  - `seg_dinov3_vitl16_ext_all.pt`
  - `dinov3_vitl16_config/config.json`
  - `restore_big_all.pt`
  - `LICENSE_DINOv3.md` (the DINOv3 License, which must accompany redistributed DINOv3 weights) and `THIRD_PARTY_LICENSES.txt`
  - SHA-256 of the archive: `bf9a8764b525e9760c80a031f8c02f830a5da68862404d13cc7d3b74ef72b668`
- The leaderboard submission was produced on a Kaggle **Tesla T4**, with segmentation under fp16 autocast (CUDA) and restoration in fp32. Another GPU may change a few mask pixels near class boundaries and the last bit of a few restored values.
- Verified on a Kaggle T4 against the #19059 zip:
  - `run.py` with `transformers==4.57.6` gives byte-identical restored images on all 1,200 images.
  - On a 40-image check, the masks are pixel-identical (agreement 1.000000), also when run with transformers 5.x and `from_pretrained`.
- Run time on a T4 is about 2.1 s/image (about 42 min for the 1,200 testing-phase images), with roughly 6 GB of GPU memory. Test images are decoded one at a time; many are full-resolution and up to tens of megapixels.
