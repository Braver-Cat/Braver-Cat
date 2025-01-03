from rich.console import RenderableType
from rich.progress import Progress
from rich.table import Table

class CustomProgress(Progress):
    def __init__(self, *args, **kwargs):
        super(CustomProgress, self).__init__(*args)

        self.val_color = kwargs["val_color"] if "val_color" in kwargs else "#ff0000"
        self.train_color = kwargs["train_color"] if "train_color" in kwargs else "#ff0000"
        self.test_color = kwargs["test_color"] if "test_color" in kwargs else "#ff0000"


    def get_renderables(self):
        if hasattr(self, "table"):
            table = self.table
        else:
            table = Table()
        task_table = self.make_tasks_table(self.tasks)
        task_table.add_row(table)
        yield task_table
    
    def update_table(self,
        running_train_loss,
        running_val_loss,
        best_train_loss,
        best_val_loss,
        delta_train_loss, 
        delta_val_loss,
        best_epoch_train_loss,
        best_epoch_val_loss,
        running_train_acc,
        running_val_acc,
        best_train_acc,
        best_val_acc,
        best_epoch_train_acc,
        best_epoch_val_acc,
        running_test_acc=None
    ):
        self.table = Table()
        self.table.add_column("Stage")
        self.table.add_column("Current Loss")
        self.table.add_column("Best Loss")
        self.table.add_column("Best Loss delta")
        self.table.add_column("Current Accuracy")
        self.table.add_column("Best Accuracy")

        self.table.add_row(
            f"[{self.train_color} bold] Train", 
            f"{running_train_loss:.3f}",
            f"{best_train_loss:.3f} [{self.train_color} bold]({best_epoch_train_loss})",
            f"{delta_train_loss:.3f}",
            f"{running_train_acc:.2f}", 
            f"{best_train_acc:.2f} [{self.train_color} bold]({best_epoch_train_acc})"
        )

        self.table.add_row(
            f"[{self.val_color} bold] Validation", 
            f"{running_val_loss:.3f}",
            f"{best_val_loss:.3f} [{self.val_color} bold]({best_epoch_val_loss})",
            f"{delta_val_loss:.3f}",
            f"{running_val_acc:.2f}", 
            f"{best_val_acc:.2f} [{self.val_color} bold]({best_epoch_val_acc})"
        )

        if running_test_acc is not None:
            self.table.add_row(
            f"[{self.test_color} bold] Test", 
            f" ♠♣ ",
            f" ♠♣ ",
            f" ♠♣ ",
            f"{running_test_acc:.2f}", 
            f"{running_test_acc:.2f} [{self.test_color} bold]({69})"
        )