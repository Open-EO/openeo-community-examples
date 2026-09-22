---
license: apache-2.0
library_name: terratorch
datasets:
  - ibm-esa-geospatial/TerraMesh
tags:
  - Earth Observation
  - TerraMind
  - IBM
  - ESA
---
[![Website](https://img.shields.io/badge/Website-TerraMind-0F62FE)](https://ibm.github.io/terramind/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.11171-b31b1b?logo=arxiv)](https://arxiv.org/abs/2504.11171)
[![Docs](https://img.shields.io/badge/Docs-EE4B2B?logo=materialformkdocs&logoColor=fff)](https://terrastackai.github.io/terratorch/stable/guide/terramind/)
[![Examples](https://img.shields.io/badge/GitHub-Examples-0F62FE?logo=github)](https://github.com/IBM/terramind)
[![Code](https://img.shields.io/badge/Code-TerraTorch-EE4B2B?logo=github)](https://github.com/terrastackai/terratorch/tree/main/terratorch/models/backbones/terramind)
[![ESAblog](https://img.shields.io/badge/Blog-ESA-113145)](https://www.esa.int/Applications/Observing_the_Earth/ESA_and_IBM_collaborate_on_TerraMind)
[![IBMblog](https://img.shields.io/badge/Blog-IBM-0F62FE)](https://research.ibm.com/blog/terramind-esa-earth-observation-model)

# TerraMind 1.0 base

TerraMind is the first multimodal any-to-any generative foundation model for Earth Observation jointly developed by IBM, ESA, and Forschungszentrum Jülich.


![terramind_architecture.png](assets%2Fterramind_architecture.png)

## Architecture

TerraMind uses a dual-scale transformer-based encoder-decoder architecture, simultaneously processing pixel-level and token-level data. 
The model was pre-trained on 500B tokens from 9M spatiotemporally aligned multimodal samples from the TerraMesh dataset.

Modality-specific patch embeddings allow direct processing of raw inputs, while modality-specific FSQ-VAEs are used for image tokenization. 
For sequence-like modalities such as coordinates, an adapted WordPiece tokenizer is employed. 
During pre-training, TerraMind leverages masked token reconstruction, learning complex cross-modal correlations to generate high-quality latent representations.

## Evaluation

![terramind_evaluation.png](assets%2Fterramind_evaluation.png)

We benchmarked TerraMind against other geospatial foundation models using the PANGAEA benchmark. 
TerraMind consistently achieved state-of-the-art performance, surpassing existing models in various downstream tasks such as land use segmentation, water body mapping, and vegetation assessments.
The evaluation highlights its effectiveness in handling diverse Earth Observation scenarios.
We present additional experiments in our [paper](https://arxiv.org/abs/2504.11171).


## Usage

TerraMind is fully integrated into the fine-tuning package [TerraTorch](https://terrastackai.github.io/terratorch/). 
This makes it easy to initialize the pre-trained model or fine-tune it via PyTorch Lightning. 
The weights are automatically downloaded from Hugging Face. 

### Fine-tuning

You can fine-tune TerraMind with a config using TerraTorch: 

```shell
terratorch fit -c terramind_config.yaml
```

For testing the fine-tuned TerraMind model, run:
```shell
terratorch test -c terramind_config.yaml --ckpt_path path/to/your/checkpoint.ckpt
```

We provide config examples and notebooks with step-by-step explanations at https://github.com/IBM/terramind.

### Backbone

Alternatively, you can build the backbone with the following code and use it in your custom pipeline.  

```python
from terratorch import BACKBONE_REGISTRY
model = BACKBONE_REGISTRY.build(
    'terramind_v1_base', 
    pretrained=True, 
    modalities=['S2L2A', 'S1GRD']    
)
```

The model supports the following raw inputs which you can specify in `modalities`: S2L2A, S2L1C, S1GRD, S1RTC, DEM, RGB. 
If your data does not use all bands of a modality, you can specify a subset with `bands={'S2L2A': ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']}`. 
You can pass the inputs as in a dict to the model. If a tensor is directly passed, the model assumes it is the first defined modality. 
TerraMind can also handle missing input modalities.

```python
output = model(
  {
    'S2L2A': s2l2a_tensor,  # B, 12, 224, 224
    'S1GRD': s1grd_tensor,  # B, 2, 224, 224
  }
)

output.shape  # B, 196, 768
```

The model outputs patch embeddings for each input modality. By default, the patch embeddings are averaged over all modalities to reduce the output size.
You can specify another `merge_method` from `'mean'`, `'max'`, `'concat'`, `'dict'`, and `None`.
- `mean` and `max` are applied per patch over all image modality embeddings.
- `concat` stacks all image modalities along the embedding dimension and returns one embedding per patch.
- `dict` returns all tokens split by modality in a dictionary. 
- `None` returns the tokens without further processing.

### Thinking in Modalities

TerraMind introduces a new Thinking-in-Modalities (TiM) approach, where other modalities are predicted as an intermediate steps.
Then, the fine-tuned encoder uses both raw inputs and the generated modalities.

Use TiM models in TerraTorch by adding `_tim` to the model name:
```python
from terratorch import BACKBONE_REGISTRY
model = BACKBONE_REGISTRY.build(
    'terramind_v1_base_tim', 
    pretrained=True, 
    modalities=['S2L2A', 'S1GRD'],
    tim_modalities=['LULC']  # optional, defaults to LULC (land-use land-cover)
)
```

If you use TiM models, we recommend using the [pre-training statistics](https://github.com/terrastackai/terratorch/blob/main/terratorch/models/backbones/terramind/model/terramind_register.py#L145) for standardization.

### Generations

TerraMind can perform any-to-any generation based on varying combinations of inputs.   

![terramind_generations.png](assets%2Fterramind_generations.png)

Build the full TerraMind model (including de-tokenizer steps) from the `FULL_MODEL_REGISTRY`:

```python
from terratorch import FULL_MODEL_REGISTRY 

model = FULL_MODEL_REGISTRY.build(
    'terramind_v1_base_generate',
    pretrained=False,
    modalities=['S2L2A'],
    output_modalities=['S1GRD', 'LULC'],
    timesteps=10,  # Define diffusion steps
    standardize=True,  # Apply standardization
)
```
Like the backbone, pass multiple modalities as a dict or a single modality as a tensor to the model which returns the generated `output_modalities` as a dict of tensors.
Note: These generations are not reconstructions but "mental images" representing how the model imagines the modality.
You can control generation details via the number of diffusion steps (`timesteps`) that you can pass to the constructor or the forward function.
By passing `standardize=True`, the pre-training standardization values are automatically applied to the input and output.

We provide an example notebook for generations at https://github.com/IBM/terramind.

## Feedback

Your feedback is invaluable to us. 
Please share it with us by starting a discussion in this HF repository or submitting an issue to [TerraMind](https://github.com/IBM/terramind) on GitHub.

## Challenge

Already working with TerraMind? Submit your use case to the [TerraMind Blue-Sky Challenge](https://huggingface.co/spaces/ibm-esa-geospatial/challenge), a bi-monthly award spotlighting the boldest, most imaginative ways using TerraMind.

## Citation

If you use TerraMind in your research, please cite the [TerraMind](https://arxiv.org/abs/2504.11171) paper.

```text
@article{jakubik2025terramind,
  title={TerraMind: Large-Scale Generative Multimodality for Earth Observation},
  author={Jakubik, Johannes and Yang, Felix and Blumenstiel, Benedikt and Scheurer, Erik and Sedona, Rocco and Maurogiovanni, Stefano and Bosmans, Jente and Dionelis, Nikolaos and Marsocci, Valerio and Kopp, Niklas and others},
  journal={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```