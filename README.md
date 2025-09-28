# Overview

- synthetic_datasets.py covers the data generation. I'm trying to include VAR(n) generation, but there's a question of whether to normalize.
- meta_optimizers.py covers manual implementations of meta optimizers. the reason I moved away from torchopt is that I wanted the learning rate to be differentiable as well, so I have to write my own loop
- losses.py covers internal window/context losses.
- func_memory_module.py has the various nn.Modules that I want to train.
- note5 has the latest stuff on 


### Outdated stuff
- notebook uses memory_module, which I don't use anymore
- notebook2 uses new_memory_module, which I also don't use anymore
- note3, 3b, and 4 use torchopt, and those aren't suitable either.
- temp is also some trash. don't need.