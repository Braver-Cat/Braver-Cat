import wandb

class WandBHelper():
    
  def __init__(
    self, project, entity, parsed_args, other_args, model
  ):
    
    self.project = project
    self.entity = entity
    self.parsed_args = parsed_args
    self.other_args = other_args
    self.model = model

    self.init_has_been_called = False

    self.run = None

  def _merge_dicts(self, dict_1, dict_2):
    merged_dicts = dict(dict_1)
    merged_dicts.update(dict_2)

    return merged_dicts

  def init_run(self):

    self.run = wandb.init(
      project=self.project, entity=self.entity, 
      config=self._merge_dicts(
        dict_1=vars(self.parsed_args), 
        dict_2=self.other_args
      ),
      job_type=self.parsed_args.job_type
    )

    self.init_has_been_called = True

  def update_config(self, config_update):
    
    wandb.config.update(config_update)

  def watch(self): 
    wandb.watch(self.model)

  def log(
    self, epoch, running_loss_train, running_loss_val, running_train_acc, 
    running_val_acc
  ):
    
    wandb.log(
      {
        "epoch": epoch,
        "loss/train": running_loss_train,
        "loss/val": running_loss_val,
        "accuracy/train": running_train_acc,
        "accuracy/val": running_val_acc,
      }
    )

  






  
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

