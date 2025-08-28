# Prepare the SCARFLightning Module
from ts3l.pl_modules import SCARFLightning
from ts3l.utils.scarf_utils import SCARFDataset
from ts3l.utils import TS3LDataModule
from ts3l.utils.scarf_utils import SCARFConfig
from ts3l.utils.embedding_utils import IdentityEmbeddingConfig
from ts3l.utils.backbone_utils import MLPBackboneConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from ts3l.models.scarf import NTXentLoss
from target_source_instance_combined import get_combined_instances

def get_scarf_encoder():
    X_combined_instances, X_instance_train, X_instance_valid = get_combined_instances()
    metric = "accuracy_score"
    input_dim = X_instance_train.shape[1]
    pretraining_head_dim = 1024
    output_dim = 2
    head_depth = 2
    dropout_rate = 0.1
    corruption_rate = 0.6
    batch_size = 64
    max_epochs = 2
    categorical_cols = ['Gender', 'City', 'Profession', 'Degree', 'Dietary Habits',
                        'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']
    continuous_cols = ['Age', 'CGPA', 'Work/Study Hours', 'Financial Stress', 'Study Satisfaction',
                        'Job Satisfaction', 'Sleep Duration', 'Work Pressure', 'Academic Pressure']

    embedding_config = IdentityEmbeddingConfig(input_dim = input_dim)
    backbone_config = MLPBackboneConfig(input_dim = embedding_config.output_dim)

    config = SCARFConfig(
                        task="classification",
                        loss_fn="CrossEntropyLoss",
                        metric=metric, metric_hparams={},
                        embedding_config=embedding_config,
                        backbone_config=backbone_config,
                        pretraining_head_dim=pretraining_head_dim,
                        output_dim=output_dim,
                        head_depth=head_depth,
                        dropout_rate=dropout_rate,
                        corruption_rate = corruption_rate
    )

    pl_scarf = SCARFLightning(config)

    ### Pre-training (Encoder Learning)
    train_ds = SCARFDataset(X_instance_train, config = config, continuous_cols=continuous_cols, category_cols=categorical_cols)
    valid_ds = SCARFDataset(X_instance_valid, config = config, continuous_cols=continuous_cols, category_cols=categorical_cols)

    datamodule = TS3LDataModule(train_ds, valid_ds, batch_size=batch_size, train_sampler="random")
    logger = CSVLogger("logs", name="scarf_first_stage")

    trainer = Trainer(
                        accelerator = 'gpu',
                        max_epochs = max_epochs,
                        num_sanity_val_steps = 2,
                        log_every_n_steps=10,
                        logger=False,
        )


    trainer.fit(pl_scarf, datamodule)
    encoder = pl_scarf.model.backbone_module.backbone
    return encoder








