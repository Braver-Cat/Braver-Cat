from rich.panel import Panel
from rich.live import Live
from rich.progress import *
from rich.console import Console, ConsoleOptions, RenderResult
from rich.layout import Layout
from rich.console import Console
from rich import print
from rich.style import Style
import asciichartpy as acp
import numpy as np
import keyboard

EPOCHS_COLOR = "#830a48"
TRAIN_COLOR = "#2a9d8f"
VAL_COLOR = "#065a82"
TEST_COLOR = "#00008B"
FINISHED_COLOR = "#3CB371"

class Dashboard():
  def __init__(self, params: dict, terminal_theme, key_closer = False):
    self.params = params
 
    self.text_color = "#000000" if terminal_theme == "white" else "white"
    self.params_color = "#DAA520" if terminal_theme == "white" else "#FED766"
    self.pbar_color = "#9400D3" if terminal_theme == "white" else "#9593D9"

    self.current_epoch = 0
    self.to_print = []
    self.max_row_print = 40

    self.train_losses = []
    self.val_losses = []
    self.train_accs = []
    self.val_accs = []

    self.live = None
    self.stopped = False
    self.key_closer = key_closer
    self.layout = Layout()
    self.current_dict = {
      "train": {
        "loss": np.inf,
        "accuracy": 0
      },
      "val": {
        "loss": np.inf,
        "accuracy": 0
      }
    }
    self.best_dict = {
      "train": {
        "loss": (np.inf, 0, -1),
        "accuracy": (-1, 0, -1)
      },
      "val": {
        "loss": (np.inf, 0, -1),
        "accuracy": (-1, 0, -1)
      }
    }

    self.progress = Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}", justify="right"),
      TaskProgressColumn(),
      BarColumn(
        bar_width=None, 
        style=Style(
          color=terminal_theme
        ),
        complete_style=Style(
          color=self.pbar_color
        ),
        finished_style=Style(
          color=FINISHED_COLOR
        )
      ),
      MofNCompleteColumn(),
      TextColumn("•"),
      TimeElapsedColumn(),
      TextColumn("•"),
      TimeRemainingColumn(),
      expand=True
    )

    self.epochs_bar = self.progress.add_task(f"[{EPOCHS_COLOR} bold]Epochs", total=self.params["n_epochs"])
    self.train_bar = self.progress.add_task(f"[{TRAIN_COLOR} bold]Train", total=self.params["train_batches"])
    self.val_bar = self.progress.add_task(f"[{VAL_COLOR} bold]Validation", total=self.params["val_batches"])
    self.test_bar = self.progress.add_task(f"[{TEST_COLOR} bold]Test", total=self.params["test_batches"])

    self._init_panels()

    if self.key_closer:
      self.init_keyboard()
  
  def println(self, out):
    self.to_print.append(str(out))
    self.layout["output"].update(Panel(Output_renderable(self.to_print), title="[bold]Logs"))
  
  def _init_panels(self):
    self.build_layout()
    self.build_progress_bars()
    self.build_params()
    self.layout["output"].update(Panel("", title="[bold]Logs"))
    self._build_stats()
    self.layout["train_loss"].update(Panel("", title="[bold]Training loss", style=TRAIN_COLOR))
    self.layout["val_loss"].update(Panel("", title="[bold]Validation loss", style=VAL_COLOR))
    self.layout["train_acc"].update(Panel("", title="[bold]Training accuracy", style=TRAIN_COLOR))
    self.layout["val_acc"].update(Panel("", title="[bold]Validation accuracy", style=VAL_COLOR))
    self.layout["test_stats"].update(Panel("", title="[bold]Test stats", style=TEST_COLOR))
    

  def build_layout(self):
    self.layout.split(
      Layout(ratio=1, name="main"),
      Layout(name="progress_bars", size=6),
    )
    self.layout["main"].split(
      Layout(name="params", size=25),
      Layout(name="train_panel", ratio=1),
      Layout(name="val_panel", ratio=1),
      Layout(name="test_out_panel", ratio=1),
      splitter="row"
    )
    self.layout["test_out_panel"].split(
      Layout(name="output"),
      Layout(name="test_stats", size=10)
    )
    self.layout["train_panel"].split(
      Layout(name="train_stats", size=10),
      Layout(name="train_graphes")
    )
    self.layout["train_graphes"].split(
      Layout(name="train_loss"),
      Layout(name="train_acc")
    )
    self.layout["val_panel"].split(
      Layout(name="val_stats", size=10),
      Layout(name="val_graphes")
    )
    self.layout["val_graphes"].split(
      Layout(name="val_loss"),
      Layout(name="val_acc")
    )
  
  def init_keyboard(self):
    keyboard.add_hotkey("space", lambda: self.stop() if not self.stopped else self.start())

  def kill(self):
    if self.key_closer:
      keyboard.remove_hotkey("space")

  ## Bar plotting stuff

  def build_progress_bars(self):
    self.layout["progress_bars"].update(Panel(self.progress, title="[bold]Progress bars", style=self.pbar_color))
  
  def epoch_step(self, train_loss, val_loss, train_acc, val_acc):
    self.current_epoch += 1
    self.progress.update(self.epochs_bar, completed=self.current_epoch)
    if self.current_epoch != self.params["n_epochs"]:
      self.progress.reset(self.train_bar)
      self.progress.reset(self.val_bar)

    self.current_dict["train"] = {"loss": train_loss, "accuracy": train_acc}
    self.current_dict["val"] = {"loss": val_loss, "accuracy": val_acc}

    self._update_bests(train_loss, val_loss, train_acc, val_acc)
    self._build_stats()

    self._plot_train_loss()
    self._plot_val_loss()
    self._plot_train_accs()
    self._plot_val_accs()
  
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
  
  def init_bests(self, starting_epoch, 
      best_epoch_train_acc, best_epoch_train_loss,
      best_train_acc, best_train_loss, delta_train_loss, 
      best_val_acc, best_val_loss, delta_val_loss, 
      best_epoch_val_acc, best_epoch_val_loss
    ):
    
    self.progress.update(self.epochs_bar, completed=starting_epoch)
    self.current_epoch = starting_epoch
    train_dict = self.best_dict["train"]
    train_dict["loss"] = (best_train_loss, delta_train_loss, best_epoch_train_loss)
    train_dict["accuracy"] = (best_train_acc, 0, best_epoch_train_acc)

    val_dict = self.best_dict["val"]
    val_dict["loss"] = (best_val_loss, delta_val_loss, best_epoch_val_loss)
    val_dict["accuracy"] = (best_val_acc, 0, best_epoch_val_acc)




  ## Show parameter

  def build_params(self):
    formatted = "\n".join([f"[bold]{key}:[/bold] {value}" for key, value in self.params.items()])
    self.layout["params"].update(Panel(formatted, title="[bold]Parameters", padding=(1), style=self.params_color))

  ## Show and update best scores
  
  def _update_bests(self, train_loss=None, val_loss=None, train_acc=None, val_acc=None):
    if train_loss != None and train_loss < self.best_dict["train"]["loss"][0]:
      delta = self.best_dict["train"]["loss"][0] - train_loss 
      self.best_dict["train"]["loss"] = (train_loss, delta, self.current_epoch)
    if val_loss != None and val_loss < self.best_dict["val"]["loss"][0]:
      delta = self.best_dict["val"]["loss"][0] - val_loss 
      self.best_dict["val"]["loss"] = (val_loss, delta, self.current_epoch)
    if train_acc != None and train_acc > self.best_dict["train"]["accuracy"][0]:
      delta = self.best_dict["train"]["accuracy"][0] - train_acc 
      self.best_dict["train"]["accuracy"] = (train_acc, delta, self.current_epoch)
    if val_acc != None and val_acc > self.best_dict["val"]["accuracy"][0]:
      delta = self.best_dict["val"]["accuracy"][0] - val_acc 
      self.best_dict["val"]["accuracy"] = (val_acc, delta, self.current_epoch)
  
  def _build_stats(self):
    train_stats = f"[{self.text_color}][bold]Current Loss:[/bold] {self.current_dict['train']['loss']:.3f}\n"
    train_stats += f"[bold]Best Loss:[/bold] {self.best_dict['train']['loss'][0]:.3f} (epoch {self.best_dict['train']['loss'][2]})\n"
    train_stats += f"[bold]Delta Loss:[/bold] {self.best_dict['train']['loss'][1]:.3f}\n\n"

    train_stats += f"[bold]Current Accuracy:[/bold] {self.current_dict['train']['accuracy']:.2f}\n"
    train_stats += f"[bold]Best Accuracy:[/bold] {self.best_dict['train']['accuracy'][0]:.2f} (epoch {self.best_dict['train']['accuracy'][2]})\n"
    train_stats += f"[bold]Delta Accuracy:[/bold] {self.best_dict['train']['accuracy'][1]:.2f}[/{self.text_color}]"

    self.layout["train_stats"].update(Panel(train_stats, title="[bold]Training stats", style=TRAIN_COLOR))

    val_stats = f"[{self.text_color}][bold]Current Loss:[/bold] {self.current_dict['val']['loss']:.3f}\n"
    val_stats += f"[bold]Best Loss:[/bold] {self.best_dict['val']['loss'][0]:.3f} (epoch {self.best_dict['val']['loss'][2]})\n"
    val_stats += f"[bold]Delta Loss:[/bold] {self.best_dict['val']['loss'][1]:.3f}\n\n"

    val_stats += f"[bold]Current Accuracy:[/bold] {self.current_dict['val']['accuracy']:.2f}\n"
    val_stats += f"[bold]Best Accuracy:[/bold] {self.best_dict['val']['accuracy'][0]:.2f} (epoch {self.best_dict['val']['accuracy'][2]})\n"
    val_stats += f"[bold]Delta Accuracy:[/bold] {self.best_dict['val']['accuracy'][1]:.2f}[/{self.text_color}]"

    self.layout["val_stats"].update(Panel(val_stats, title="[bold]Validation stats", style=VAL_COLOR))

  # Plotting graph

  def _plot_train_loss(self):
    self.train_losses.append(self.current_dict["train"]["loss"])
    self.layout["train_loss"].update(Panel(Graph_renderable(self.train_losses), title="[bold]Training loss", style=TRAIN_COLOR))

  def _plot_val_loss(self):
    self.val_losses.append(self.current_dict["val"]["loss"])
    self.layout["val_loss"].update(Panel(Graph_renderable(self.val_losses), title="[bold]Validation loss", style=VAL_COLOR))
  
  def _plot_val_accs(self):
    self.val_accs.append(self.current_dict["val"]["accuracy"])
    self.layout["val_acc"].update(Panel(Graph_renderable(self.val_accs), title="[bold]Validation accuracy", style=VAL_COLOR))
  
  def _plot_train_accs(self):
    self.train_accs.append(self.current_dict["train"]["accuracy"])
    self.layout["train_acc"].update(Panel(Graph_renderable(self.train_accs), title="[bold]Training accuracy", style=TRAIN_COLOR))
  
  
  def start(self):
    self.stopped = False
    if self.live != None:
      self.live.start()
      return False
    self.live = Live(self.layout, screen=True)
    self.live.start()
    return True

  def stop(self):
    if self.live == None:
      return False
    self.live.stop()
    self.live.console.clear()
    self.stopped = True
    return self.live.renderable

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