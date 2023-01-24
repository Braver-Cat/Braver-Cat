import wandb

class WandBHelper():
    
  def __init__(self, project, entity, parsed_args):
    
    self.project = project
    self.entity = entity
    self.parsed_args = parsed_args

    self.init_has_been_called = False

  def init_run(self):

    wandb.init(
      project=self.project, entity=self.entity, config=self.parsed_args,
      job_type=self.parsed_args.job_type
    )

    self.init_has_been_called = True

  






  
  # def set_wandb_config(self, parsed_args):

  #   wandb.config = {
  #     "dataset_df_path": parsed_args.dataset_df_path,
  #     "patch_size": parsed_args.patch_size,
  #     "batch_size": parsed_args.batch_size,
  #     "cascade_type": parsed_args.cascade_type,
  #     "optimizer": parsed_args.optimizer_name,
  #     "momentum": parsed_args.momentum,
  #     "learning_rate": parsed_args.learning_rate,
  #     "learning_rate_decay_factor": parsed_args.learning_rate_decay_factor,
  #     "learning_rate_decay_step_size": parsed_args.learning_rate_decay_step_size,
  #     "num_batches": parsed_args.num_batches,
  #     "delta_1": parsed_args.delta_1,
  #     "delta_2": parsed_args.delta_2,
  #     "dropout": parsed_args.dropout,
  #     "num_epochs": parsed_args.num_epochs,
  #   }

