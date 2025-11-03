from dataclasses import dataclass, field


@dataclass
class Arguments:
    dataset_path: str = field(default="./datasets/ChnSentiCorp_htl_all.csv")
    model_name_or_path: str = field(default="./models/rbt3")
    output_dir: str = field(default="./results/text_classification_v3")
    num_epoch: int = field(default=3)
    log_step: int = field(default=10)
    lr: float = field(default=2e-5)
