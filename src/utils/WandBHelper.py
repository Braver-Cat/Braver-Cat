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
        dict_1=self.parsed_args, 
        dict_2=self.other_args
      ),
      job_type=self.parsed_args["job_type"]
    )

    self.init_has_been_called = True

  def update_config(self, config_update):
    
    wandb.config.update(config_update)

  def watch(self): 
    wandb.watch(self.model)

  def log(
    self, epoch, running_loss_train, running_loss_val, running_train_acc, 
    running_val_acc, learning_rate
  ):
    
    wandb.log(
      {
        "epoch": epoch,
        "loss/train": running_loss_train,
        "loss/val": running_loss_val,
        "accuracy/train": running_train_acc,
        "accuracy/val": running_val_acc,
        "train/lr": learning_rate
      }
    )

  def get_run_url(self):
    return wandb.run.get_url()
  
  