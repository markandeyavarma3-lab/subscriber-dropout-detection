"""Loaders that replace the simulator with real subscriber data.

Everything else in this project is exercised against
:mod:`src.warehouse.simulate`, which is honest about what it is: a generator
whose churn hazard we wrote. The machinery around it - point-in-time features,
temporal splits, promotion gates, drift detection - is real and tested, but no
number it produces is evidence about real subscribers.

This package is the seam where that changes. A loader's whole job is to turn
somebody else's export into the five warehouse tables; nothing downstream of
:mod:`src.warehouse.schema` knows or cares which loader filled them.
"""
