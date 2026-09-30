# Data folder

## Where to put the Kaggle files

1. Download the **Mobile Price Classification** dataset:
   https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
2. Place the files here so the paths match the code:
   - `data/train.csv`  (must contain all 20 feature columns + `price_range`)
   - `data/test.csv`   (optional; Kaggle test set has no `price_range` column)

Expected columns in `train.csv`:

battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w,
talk_time, three_g, touch_screen, wifi, price_range

## Current status

`train.csv` in this folder was downloaded from a public GitHub mirror of the
exact same Kaggle dataset (2000 rows, balanced 500 per class) so the pipeline
can be executed immediately:
https://github.com/arpita-maji/Mobile-Price-Classification/blob/master/mobile_price_range_data.csv

If you want the pristine Kaggle copy, just overwrite `data/train.csv` with it
and re-run `python -m src.train`. `test.csv` is optional and not needed for
training/evaluation.
