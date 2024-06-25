from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from typing import Optional

@dataclass(slots=True)
class Modes:
    phi1: jnp.ndarray
    psi1: jnp.ndarray
    phi2: jnp.ndarray
    phi1l: jnp.ndarray
    phi1ml: jnp.ndarray
    psi1l: jnp.ndarray
    phi2l: jnp.ndarray
    psi2l: jnp.ndarray
    omega: jnp.ndarray    
    X_xdelta: jnp.ndarray
    C0ab: jnp.ndarray
    C06ab: jnp.ndarray

@dataclass(slots=True)
class Couplings:
    alpha1: jnp.ndarray
    alpha2: jnp.ndarray
    gamma1: jnp.ndarray
    gamma2: jnp.ndarray

@dataclass(slots=True)
class DynamicSystem:
    q: jnp.ndarray
    X1: jnp.ndarray
    X2: jnp.ndarray
    X3: jnp.ndarray
    Cab: jnp.ndarray
    ra: jnp.ndarray
    t: jnp.ndarray = None
    
@dataclass(slots=True)
class StaticSystem:
    q: jnp.ndarray
    X2: jnp.ndarray = None
    X3: jnp.ndarray = None
    Cab: jnp.ndarray = None
    ra: jnp.ndarray = None

@dataclass(slots=True)
class ModalAeroRoger:

    poles: jnp.ndarray = None
    A0: jnp.ndarray = None
    A1: jnp.ndarray = None
    A2: jnp.ndarray = None
    A3: jnp.ndarray = None
    B0: jnp.ndarray = None
    B1: jnp.ndarray = None
    B2: jnp.ndarray = None
    B3: jnp.ndarray = None
    D0: jnp.ndarray = None
    D1: jnp.ndarray = None
    D2: jnp.ndarray = None
    D3: jnp.ndarray = None
    C0: jnp.ndarray = None
    A0hat: jnp.ndarray = None
    A1hat: jnp.ndarray = None
    A2hat: jnp.ndarray = None
    A2hatinv: jnp.ndarray = None
    A3hat: jnp.ndarray = None  # NlxNmxNm
    B0hat: jnp.ndarray = None
    B1hat: jnp.ndarray = None
    B2hat: jnp.ndarray = None
    B3hat: jnp.ndarray = None
    D0hat: jnp.ndarray = None
    D1hat: jnp.ndarray = None
    D2hat: jnp.ndarray = None
    D3hat: jnp.ndarray = None  # NlxNmxNb
    C0hat: jnp.ndarray = None
    
@dataclass(slots=True)
class GustRoger:

    w: jnp.ndarray = None
    wdot: jnp.ndarray = None
    wddot: jnp.ndarray = None
    x: jnp.ndarray = None
    Qhj_w:jnp.ndarray = None
    Qhj_wdot:jnp.ndarray = None
    Qhj_wddot:jnp.ndarray = None
    Qhj_wsum:jnp.ndarray = None
    Qhjl_wdot:jnp.ndarray = None

@dataclass(slots=True)
class ModalAeroStatespace:
    
    A: Optional[jnp.ndarray] = None
    B0: Optional[jnp.ndarray] = None
    B1: Optional[jnp.ndarray] = None
    Bw: Optional[jnp.ndarray] = None
    C: Optional[jnp.ndarray] = None
    D0: Optional[jnp.ndarray] = None
    D1: Optional[jnp.ndarray] = None
    Dw: Optional[jnp.ndarray] = None
    Ahat: Optional[jnp.ndarray] = None
    B0hat: Optional[jnp.ndarray] = None
    B1hat: Optional[jnp.ndarray] = None
    Bwhat: Optional[jnp.ndarray] = None
    Chat: Optional[jnp.ndarray] = None
    D0hat: Optional[jnp.ndarray] = None
    D1hat: Optional[jnp.ndarray] = None
    Dwhat: Optional[jnp.ndarray] = None

@dataclass(slots=True)
class GustStatespace:

    w: Optional[jnp.ndarray] = None
    x: Optional[jnp.ndarray] = None
    Bw_gust: Optional[jnp.ndarray] = None
    Dw_gust: Optional[jnp.ndarray] = None

# import dataclasses    
# field_types = {field.name: field.type for field in dataclasses.fields(Modes)}
