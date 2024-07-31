from typing import Any

import numpy as np
import numpy.typing as npt

class rv_generic: ...

class rv_continuous(rv_generic):
    def __init__(
        self,
        momtype: Any = 1,
        a: Any = None,
        b: Any = None,
        xtol: Any = 1e-14,
        badvalue: Any = None,
        name: Any = None,
        longname: Any = None,
        shapes: Any = None,
        seed: Any = None,
    ): ...

class norm_gen(rv_continuous):
    def cdf(self, x: Any, *args: Any, **kwds: Any) -> npt.NDArray[np.float64]: ...

norm = norm_gen(name="norm")
