# ITL-LIME: Instance-Based Transfer Learning for Enhancing Local Explanations in Low-Resource Data Settings

A novel Instance-based Transfer Learning LIME framework (ITL-LIME) that enhances explanation fidelity and stability in dataconstrained environments. ITL-LIME introduces instance transfer learning into the LIME framework by leveraging relevant real instances from a related source domain to aid the explanation process
in the target domain. Specifically, it employ clustering to partition the source domain into clusters with representative prototypes. Instead of generating random perturbations, ITL-LIME method retrieves pertinent real source instances from the source cluster whose prototype is most similar to the target instance. 
These are then combined with the target instance’s neighboring real instances. To define a compact locality, ITL-LIME further construct a contrastive learning based encoder as a weighting mechanism to assign weights to the instances from the combined set based on their proximity to the target instance. 
Finally, these weighted source and target instances are used to train the surrogate model for explanation purposes. 

## Installation
```sh
pip install ts3l
```

## SCARF Github
Link: https://github.com/Alcoholrithm/TabularS3L

#### SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption
SCARF introduces a contrastive learning framework specifically tailored for tabular data. By corrupting random subsets of features, SCARF creates diverse views for self-supervised learning in its first phase. The subsequent phase transitions to supervised learning, utilizing a pretrained encoder to enhance model accuracy and robustness.

<details close>
  <summary>Quick Start</summary>
  
  ```python
  # Assume that we have X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_cols, and continuous_cols

  # Prepare the SCARFLightning Module
  from ts3l.pl_modules import SCARFLightning
  from ts3l.utils.scarf_utils import SCARFDataset
  from ts3l.utils import TS3LDataModule
  from ts3l.utils.scarf_utils import SCARFConfig
  from ts3l.utils.embedding_utils import IdentityEmbeddingConfig
  from ts3l.utils.backbone_utils import MLPBackboneConfig
  from pytorch_lightning import Trainer

  metric = "accuracy_score"
  input_dim = X_train.shape[1]
  pretraining_head_dim = 1024
  output_dim = 2
  head_depth = 2
  dropout_rate = 0.04

  corruption_rate = 0.6

  batch_size = 128
  max_epochs = 10

  X_train, X_unlabeled, y_train, _ = train_test_split(X_train, y_train, train_size = 0.1, random_state=0, stratify=y_train)

  embedding_config = IdentityEmbeddingConfig(input_dim = input_dim)
  backbone_config = MLPBackboneConfig(input_dim = embedding_config.output_dim)

  config = SCARFConfig( 
                      task="classification", loss_fn="CrossEntropyLoss", metric=metric, metric_hparams={},
                      embedding_config=embedding_config, backbone_config=backbone_config,
                      pretraining_head_dim=pretraining_head_dim,
                      output_dim=output_dim, head_depth=head_depth,
                      dropout_rate=dropout_rate, corruption_rate = corruption_rate
  )

  pl_scarf = SCARFLightning(config)

  ### First Phase Learning
  train_ds = SCARFDataset(X_train, unlabeled_data=X_unlabeled, config = config, continuous_cols=continuous_cols, category_cols=category_cols)
  valid_ds = SCARFDataset(X_valid, config=config, continuous_cols=continuous_cols, category_cols=category_cols)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size=batch_size, train_sampler="random")

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )

  trainer.fit(pl_scarf, datamodule)

  ### Second Phase Learning

  pl_scarf.set_second_phase()

  train_ds = SCARFDataset(X_train, y_train.values, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
  valid_ds = SCARFDataset(X_valid, y_valid.values, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)

  datamodule = TS3LDataModule(train_ds, valid_ds, batch_size = batch_size, train_sampler="weighted")

  trainer = Trainer(
                      accelerator = 'cpu',
                      max_epochs = max_epochs,
                      num_sanity_val_steps = 2,
      )

  trainer.fit(pl_scarf, datamodule)

  # Evaluation
  from sklearn.metrics import accuracy_score
  import torch
  from torch.nn import functional as F
  from torch.utils.data import DataLoader, SequentialSampler

  test_ds = SCARFDataset(X_test, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
  test_dl = DataLoader(test_ds, batch_size, shuffle=False, sampler = SequentialSampler(test_ds), num_workers=4)

  preds = trainer.predict(pl_scarf, test_dl)
          
  preds = F.softmax(torch.concat([out.cpu() for out in preds]).squeeze(),dim=1)

  accuracy = accuracy_score(y_test, preds.argmax(1))

  print("Accuracy %.2f" % accuracy)
  ```

</details>

## Comparison Studies

- [D-LIME Paper] (https://arxiv.org/abs/1906.10263)
- [A-LIME Paper] (https://link.springer.com/chapter/10.1007/978-3-030-33607-3_49)
- [US-LIME Paper] ([ALIME Paper])
- [Bay-LIME Paper] (https://arxiv.org/abs/2012.03058)
- [S-LIME Paper] (https://dl.acm.org/doi/abs/10.1145/3447548.3467274)


## Using the code
Have a look at the LICENSE.

## Contributing
Contributions to this implementation are highly appreciated. Whether it's suggesting improvements, reporting bugs, or proposing new features, feel free to open an issue or submit a pull request.

## Citation
If you find our work helpful in your research, please cite it as:

```
@article{raza2025itl,
  title={ITL-LIME: Instance-Based Transfer Learning for Enhancing Local Explanations in Low-Resource Data Settings},
  author={Raza, Rehan and Wang, Guanjin and Wong, Kevin and Laga, Hamid and Fisichella, Marco},
  journal={arXiv preprint arXiv:2508.13672},
  year={2025}
}
```
