from src.optimizers.pso import run_pso
from src.optimizers.ga import run_ga
from src.optimizers.aco import run_aco
from src.optimizers.woa import run_woa
from src.optimizers.bhho import run_bhho

def run_optimizer(name, **kwargs):

    if name == "pso":
        return run_pso(**kwargs)
    elif name == "ga":
        return run_ga(**kwargs)
    elif name == "aco":
        return run_aco(**kwargs)
    elif name == "woa":
        return run_woa(**kwargs)
    elif name == "bhho":
        return run_bhho(**kwargs)
    else:
        raise ValueError("Unknown optimizer")