from datasets.topological import DataModule, DataModuleConfig 

import cProfile

dm = DataModule(DataModuleConfig(pin_memory=False,num_workers=0))

print(len(dm.train_ds))
def main():
    for i in range(10):
        print(i)
        for batch in dm.train_dataloader():
            batch.cuda()

cProfile.run('main()','rstats.prof')
import pstats
from pstats import SortKey
p = pstats.Stats('rstats.prof')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats(15)

