
from Dashboard import Dashboard
import torch

import time

class TestDashboard:
  def test(self):
    dashboard = Dashboard({
      "n_epochs": 1,
      "train_batches": 1,
      "val_batches": 1,
      "test_batches": 1
    }, key_closer=False)
  
    dashboard.start()
    time.sleep(10)
    dashboard.stop()

TestDashboard().test()