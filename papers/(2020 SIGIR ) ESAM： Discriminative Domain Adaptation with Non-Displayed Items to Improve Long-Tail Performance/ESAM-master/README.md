# Updating, detailed process and explanation are coming soon

## For Long-tail Recommendation Training:
```
python test_movielens.py
```
## Data Labels(Pseudo-Labels) Preparation
- For positive samples, besides the positive items from the records, each user randomly chooses a hot item and a long-tail item.
- For negative samples, each user randomly chooses 30 hot items, 30 long-tail items and 30 negative items respectively.
