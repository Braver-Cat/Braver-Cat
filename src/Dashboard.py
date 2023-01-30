from rich.panel import Panel
from rich.live import Live
from rich.progress import *
from rich.console import Console, ConsoleOptions, RenderResult
from rich.layout import Layout
from rich.console import Console
import asciichartpy as acp

class Dashboard():
  def __init__(self, params: dict):
    self.current_epoch = 0
    self.to_print = []
    self.max_row_print = 40

    self.train_losses = []
    self.val_losses = []

    self.layout = Layout()
    self.best_dict = {
      "train": {
        "loss": (-1, -1),
        "accuracy": (-1, -1)
      },
      "val": {
        "loss": (-1, -1),
        "accuracy": (-1, -1)
      }
    }

    self.progress = Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}", justify="right"),
      TaskProgressColumn(),
      BarColumn(bar_width=None),
      MofNCompleteColumn(),
      TextColumn("•"),
      TimeElapsedColumn(),
      TextColumn("•"),
      TimeRemainingColumn(),
      expand=True
    )

    self.epochs_bar = self.progress.add_task("[red]Epochs", total=self.params["n_epochs"])
    self.train_bar = self.progress.add_task("[red]Train", total=self.params["train_batches"])
    self.val_bar = self.progress.add_task("[green]Validation", total=self.params["val_batches"])
    self.test_bar = self.progress.add_task("[cyan]Test", total=self.params["test_batches"])

    self.build_layout()
    self.build_params()
    self.build_progress_bars()
    self.build_bests()
    self.init_out_panel()
  
  def println(self, out):
    self.to_print.append(str(out))
    self.layout["output"].update(Panel(Output_renderable(self.to_print)))
  
  def init_out_panel(self):
    self.layout["output"].update(Panel(""))

  def build_layout(self):
    self.layout.split(
      Layout(ratio=1, name="main"),
      Layout(name="progress_bars", size=6),
    )
    self.layout["main"].split(
      Layout(name="side", size=30),
      Layout(name="body", ratio=4),
      splitter="row"
    )
    self.layout["side"].split(
      Layout(name="params"),
      Layout(name="bests", size=8)
    )
    self.layout["body"].split(
      Layout(name="graphes"),
      Layout(name="output", size=30),
      splitter="row"
    )
    self.layout["graphes"].split(
      Layout(name="col1"),
      Layout(name="col2"),
      splitter="row"
    )
    self.layout["col1"].split(
      Layout(name="train_loss"),
      Layout(name="val_loss")
    )
    
  
  ## Bar plotting stuff

  def build_progress_bars(self):
    self.layout["progress_bars"].update(Panel(self.progress, title="Progress bars"))
  
  def epoch_step(self):
    self.current_epoch += 1
    self.progress.update(self.epochs_bar, completed=self.current_epoch)
    if self.current_epoch != self.params["n_epochs"]:
      self.progress.reset(self.train_bar)
      self.progress.reset(self.val_bar)
  
  def train_step(self):
    epoch_advance = 1 / (self.params["train_batches"] + self.params["val_batches"])
    self.progress.advance(self.train_bar)
    self.progress.advance(self.epochs_bar, epoch_advance)
  
  def val_step(self):
    epoch_advance = 1 / (self.params["train_batches"] + self.params["val_batches"])
    self.progress.advance(self.val_bar)
    self.progress.advance(self.epochs_bar, epoch_advance)

  def test_step(self):
    self.progress.advance(self.test_bar)

  ## Show parameter

  def build_params(self):
    formatted = "\n".join([f"{key}: {value}" for key, value in self.params.items()])
    self.layout["params"].update(Panel(formatted, title="Parameters", padding=(1)))

  ## Show and update best scores
  
  def build_bests(self, train_loss=None, val_loss=None, train_acc=None, val_acc=None):
    if train_loss != None:
      self.best_dict["train"]["loss"] = (train_loss, self.current_epoch)
    if val_loss != None:
      self.best_dict["val"]["loss"] = (val_loss, self.current_epoch)
    if train_acc != None:
      self.best_dict["train"]["accuracy"] = (train_acc, self.current_epoch)
    if val_acc != None:
      self.best_dict["val"]["accuracy"] = (val_acc, self.current_epoch)
    
    train_loss_lst = self.best_dict["train"]["loss"]
    val_loss_lst = self.best_dict["val"]["loss"]
    train_acc_lst = self.best_dict["train"]["accuracy"]
    val_acc_lst = self.best_dict["val"]["accuracy"]
    best_str = f"Train loss: {train_loss_lst[0]} ({train_loss_lst[1]})\nValidation loss: {val_loss_lst[0]} ({val_loss_lst[1]})\nTrain acc: {train_acc_lst[0]} ({train_acc_lst[1]})\nValidation acc: {val_acc_lst[0]} ({val_acc_lst[1]})"
    self.layout["bests"].update(Panel(best_str, title="Best scores", padding=(1)))

  # Plotting graph

  def plot_train_loss(self, loss):
    self.train_losses.append(loss)
    self.layout["train_loss"].update(Panel(Graph_renderable(self.train_losses)))

  def plot_val_loss(self, loss):
    self.val_losses.append(loss)
    self.layout["val_loss"].update(Panel(Graph_renderable(self.val_losses)))
  
  
  def start(self):
    if self.live != None:
      return False
    self.live = Live(self.layout, screen=True)
    self.live.start()
    return True

  def stop(self):
    if self.live == None:
      return False
    self.live.console.clear()
    self.live.stop()
    return True

@dataclass
class Output_renderable:
  lst: list
  
  def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
    self.lst = self.lst[len(self.lst) - options.max_height if len(self.lst) > options.max_height else 0:]
    yield "\n".join(self.lst)

@dataclass
class Graph_renderable:
  values: list

  def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
    self.values = self.values[len(self.values) + 11 - options.max_width if len(self.values) + 11 > options.max_width else 0:]
    yield acp.plot(self.values, {"height":options.max_height - 2})