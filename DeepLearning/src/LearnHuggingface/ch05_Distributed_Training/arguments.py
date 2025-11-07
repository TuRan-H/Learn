import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Arguments:
    project_name: str = field(default="text_classification_v3")
    dataset_path: str = field(default="./datasets/ChnSentiCorp_htl_all.csv")
    model_name_or_path: str = field(default="./models/rbt3")
    output_dir: str = field(default="./results/text_classification_v3")
    log_dir: str = field(default="./logs/text_classification_v3")
    run_name: str = field(default="")
    num_epoch: int = field(default=3)
    log_step: int = field(default=10)
    lr: float = field(default=2e-5)
    batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=2)
    save_steps: int = field(default=20)

    def __post_init__(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_dir, current_time)
        self.run_name = current_time

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
