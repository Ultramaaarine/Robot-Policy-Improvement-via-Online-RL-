
from typing import Union,Optional
from diffusers.optimization import (
     SchedulerType, 
    Optimizer,
    TYPE_TO_SCHEDULER_FUNCTION
)



def get_scheduler(name:Union[str, SchedulerType],
                  optimizer: Optimizer,
                  num_training_steps: Optional[int] = None,
                  num_warmup_steps: Optional[int] = None,
                  **kwargs
    ):
    name = SchedulerType(name)
    scheduler_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:

        return scheduler_func(optimizer,**kwargs)
    if name == SchedulerType.CONSTANT_WITH_WARMUP:

        return scheduler_func(optimizer,num_warmup_steps,**kwargs)
    
    # for other schedulers which reqire training steps
    if num_training_steps == None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.") 
        
    return scheduler_func(optimizer,num_training_steps,num_warmup_steps,**kwargs) 

