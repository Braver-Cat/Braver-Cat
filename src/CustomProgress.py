from rich.console import RenderableType
from rich.progress import Progress
from rich.table import Table
from collections import Iterable

class CustomProgress(Progress):
    def __init__(self, *args, **kwargs):
        super(CustomProgress, self).__init__(*args, **kwargs)
        self.val_col = kwargs["val_col"] if "val_col" in kwargs else "#065a82"
        self.train_col = kwargs["train_col"] if "train_col" in kwargs else "#2a9d8f"


    def get_renderables(self) -> Iterable[RenderableType]:
        if hasattr(self, "table"):
            table = self.table
        else:
            table = Table()
        task_table = self.make_tasks_table(self.tasks)
        task_table.add_row(table)
        yield task_table
    
    def update_table(self,
        current_train_loss,
        current_val_loss,
        best_train_loss,
        best_val_loss,
        best_epoch_train_loss,
        best_epoch_val_loss,
        current_train_acc,
        current_val_acc,
        best_train_acc,
        best_val_acc,
        best_epoch_train_acc,
        best_epoch_val_acc
    ):
        self.table = Table()
        self.table.add_column("Stage")
        self.table.add_column("Current Loss")
        self.table.add_column("Best Loss")
        self.table.add_column("Current Accuracy")
        self.table.add_column("Best Accuracy")

        self.table.add_row(
            "[bold] Train", f"{current_train_loss:.3f}",
            f"{best_train_loss:.3f} [{self.train_col} bold]({best_epoch_train_loss})",
            f"{current_train_acc:.3f}", f"{best_train_acc:.3f} [{self.train_col} bold]({best_epoch_train_acc})"
        )

        self.table.add_row(
            "[bold] Validation", f"{current_val_loss:.3f}",
            f"{best_val_loss:.3f} [{self.val_col} bold]({best_epoch_val_loss})",
            f"{current_val_acc:.3f}", f"{best_val_acc:.3f} [{self.val_col} bold]({best_epoch_val_acc})"
        )