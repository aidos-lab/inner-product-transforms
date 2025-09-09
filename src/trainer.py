import pydantic


class TrainerConfig(pydantic.BaseModel):
    module: str
    seed: int
    max_epochs: int
    log_every_n_steps: int
    accelerator: str
    precision: str | None
    checkpoint_interval: int
