import jax.numpy as jnp
import fem4inas.preprocessor.solution as solution
import fem4inas.preprocessor.containers.intrinsicmodal as intrinsicmodal
import typing

def _args_diffrax(input1):

    return input1

def _args_scipy(input1):

    return (input1,)

def _args_jax(self, input1):

    return input1

def _args_runge_kutta(input1):

    return input1

def catter2library(fun: typing.Callable):

    def wrapper(*args, **kwargs):

        args_ = fun(*args, **kwargs)
        solver_library = getattr(args[1],
                                 "solver_library")
        args_new = globals()[f"_args_{solver_library}"](args_)
        return args_new
    return wrapper 

############################################
@catter2library
def arg_10G1(sol: solution.IntrinsicSolution,
             system: intrinsicmodal.Dsystem,
             fem: intrinsicmodal.Dfem,
             t: float,
             *args, **kwargs):

    phi1l = sol.data.modes.phi1l
    psi2l = sol.data.modes.psi2l 
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_gravity = system.xloads.force_gravity
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma2, omega, phi1l, psi2l,
            x, force_gravity,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father, t)

@catter2library
def arg_10g11(sol: solution.IntrinsicSolution,
              system: intrinsicmodal.Dsystem,
              fem: intrinsicmodal.Dfem,
              t: float,
              *args, **kwargs):

    gamma2 = sol.data.couplings.gamma2
    phi1 = sol.data.modes.phi1l
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower
    return (gamma2, omega, phi1, x,
            force_follower, t)

@catter2library
def arg_10g121(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               fem: intrinsicmodal.Dfem,
               t: float,
               *args, **kwargs):

    phi1l = sol.data.modes.phi1l
    psi2l = sol.data.modes.psi2l 
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_dead = system.xloads.force_dead
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma2, omega, phi1l, psi2l,
            x, force_dead,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father, t)

@catter2library
def arg_10G121(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               fem: intrinsicmodal.Dfem,
               t: float,
               *args, **kwargs):

    phi1l = sol.data.modes.phi1l
    psi2l = sol.data.modes.psi2l 
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_dead = system.xloads.force_dead
    force_gravity = system.xloads.force_gravity
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma2, omega, phi1l, psi2l,
            x, force_dead, force_gravity,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father, t)

@catter2library
def arg_10g15(sol: solution.IntrinsicSolution,
              sys: intrinsicmodal.Dsystem,
              fem: intrinsicmodal.Dfem,
              t: float,
              *args, **kwargs):

    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    qalpha = sys.aero.qalpha
    aero = getattr(sol.data, f"modalaeroroger_{sys.name}")
    return (gamma2, omega,
            qalpha, aero.A0hat, aero.C0hat)

#########################################################
@catter2library
def arg_20g1(sol: solution.IntrinsicSolution,
             system: intrinsicmodal.Dsystem,
             *args, **kwargs):

    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    states = system.states
    return gamma1, gamma2, omega, states

@catter2library
def arg_20g11(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               *args, **kwargs):

    phi1 = sol.data.modes.phi1l
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower    
    states = system.states
    return (gamma1, gamma2, omega, phi1,
            x, force_follower, states)

@catter2library
def arg_20g121(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               fem: intrinsicmodal.Dfem,
               *args, **kwargs):

    phi1l = sol.data.modes.phi1l
    psi2l = sol.data.modes.psi2l
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    states = system.states
    force_dead = system.xloads.force_dead
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma1, gamma2, omega, phi1l, psi2l,
            x, force_dead,
            states,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father)

@catter2library
def arg_20g22(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               *args, **kwargs):

    phi1 = sol.data.modes.phi1l
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower    
    states = system.states
    return (gamma1, gamma2, omega, phi1,
            x, force_follower, states)

@catter2library
def arg_20G2(sol: solution.IntrinsicSolution,
             system: intrinsicmodal.Dsystem,
             fem: intrinsicmodal.Dfem,
             *args, **kwargs):

    phi1l = sol.data.modes.phi1l
    psi2l = sol.data.modes.psi2l
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
#     x = system.xloads.x
    states = system.states
    force_gravity = system.xloads.force_gravity
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma1, gamma2, omega, phi1l, psi2l,
            force_gravity,
            states,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father)

@catter2library
def arg_20g242(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               fem: intrinsicmodal.Dfem,
               *args, **kwargs):

    phi1l = sol.data.modes.phi1l
    psi2l = sol.data.modes.psi2l
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    states = system.states
    force_dead = system.xloads.force_dead
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma1, gamma2, omega, phi1l, psi2l,
            x, force_dead,
            states,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father)

@catter2library
def arg_20g21(sol: solution.IntrinsicSolution,
              sys: intrinsicmodal.Dsystem,
              fem: intrinsicmodal.Dfem,
              *args, **kwargs):

    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    states = sys.states
    u_inf = sys.aero.u_inf
    c_ref = sys.aero.c_ref
    num_modes = fem.num_modes
    aero = getattr(sol.data, f"modalaeroroger_{sys.name}")
    gust = getattr(sol.data, f"gustroger_{sys.name}")
    num_poles = sys.aero.num_poles
    F1g = gust.Qhj_wsum  # NmxNt
    Flg = gust.Qhjl_wdot  # NpxNmxNt (NumPoles_NumModes_NumTime)
    return (gamma1, gamma2, omega, states,
            num_modes, num_poles,
            aero.A0hat, aero.A1hat, aero.A2hatinv, aero.A3hat,
            u_inf, c_ref, aero.poles,
            gust.x, F1g, Flg)

@catter2library
def arg_20g21l(sol: solution.IntrinsicSolution,
               sys: intrinsicmodal.Dsystem,
               fem: intrinsicmodal.Dfem,
               *args, **kwargs):

    omega = sol.data.modes.omega
    states = sys.states
    u_inf = sys.aero.u_inf
    c_ref = sys.aero.c_ref
    num_modes = fem.num_modes
    aero = getattr(sol.data, f"modalaeroroger_{sys.name}")
    gust = getattr(sol.data, f"gustroger_{sys.name}")
    num_poles = sys.aero.num_poles
    F1g = gust.Qhj_wsum  # NmxNt
    Flg = gust.Qhjl_wdot  # NpxNmxNt (NumPoles_NumModes_NumTime)
    return (omega, states,
            num_modes, num_poles,
            aero.A0hat, aero.A1hat, aero.A2hatinv, aero.A3hat,
            u_inf, c_ref, aero.poles,
            gust.x, F1g, Flg)

@catter2library
def arg_20g273(sol: solution.IntrinsicSolution,
              sys: intrinsicmodal.Dsystem,
              fem: intrinsicmodal.Dfem,
              *args, **kwargs):

    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    states = sys.states
    u_inf = sys.aero.u_inf
    c_ref = sys.aero.c_ref
    num_modes = fem.num_modes
    aero = getattr(sol.data, f"modalaeroroger_{sys.name}")
    gust = getattr(sol.data, f"gustroger_{sys.name}")
    num_poles = sys.aero.num_poles
    F1g = gust.Qhj_wsum  # NmxNt
    Flg = gust.Qhjl_wdot  # NpxNmxNt (NumPoles_NumModes_NumTime)
    return (gamma1, gamma2, omega, states,
            num_modes, num_poles,
            aero.A0hat, aero.A1hat, aero.A2hatinv, aero.A3hat,
            u_inf, c_ref, aero.poles,
            gust.x, F1g, Flg)

############################################
@catter2library
def arg_20G1(sol: solution.IntrinsicSolution,
             system: intrinsicmodal.Dsystem,
             fem: intrinsicmodal.Dfem,
             *args, **kwargs):

    phi1l = sol.data.modes.phi1l
    psi2l = sol.data.modes.psi2l
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    states = system.states
    force_gravity = system.xloads.force_gravity
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma1, gamma2, omega, phi1l, psi2l,
            force_gravity,
            states,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father)

@catter2library
def arg_20g3(sol: solution.IntrinsicSolution,
              sys: intrinsicmodal.Dsystem,
              fem: intrinsicmodal.Dfem,
              *args, **kwargs):

        gamma1 = sol.data.couplings.gamma1
        gamma2 = sol.data.couplings.gamma2
        omega = sol.data.modes.omega
        states = sys.states
        u_inf = sys.aero.u_inf
        c_ref = sys.aero.c_ref
        num_modes = fem.num_modes
        aero = getattr(sol.data, f"modalaeroroger_{sys.name}")
        num_poles = sys.aero.num_poles
        return (gamma1, gamma2, omega, states,
                num_modes, num_poles,
                aero.A0hat, aero.A1hat, aero.A2hatinv, aero.A3hat,
                u_inf, c_ref, aero.poles)

@catter2library
def arg_20G3(sol: solution.IntrinsicSolution,
              sys: intrinsicmodal.Dsystem,
              fem: intrinsicmodal.Dfem,
              *args, **kwargs):

        phi1l = sol.data.modes.phi1l
        psi2l = sol.data.modes.psi2l
        gamma1 = sol.data.couplings.gamma1
        gamma2 = sol.data.couplings.gamma2
        omega = sol.data.modes.omega
        states = sys.states
        u_inf = sys.aero.u_inf
        c_ref = sys.aero.c_ref
        num_modes = fem.num_modes
        force_gravity = sys.xloads.force_gravity
        X_xdelta = sol.data.modes.X_xdelta
        C0ab = sol.data.modes.C0ab

        num_nodes = fem.num_nodes
        component_nodes = fem.component_nodes_int
        component_names = fem.component_names_int
        component_father = fem.component_father_int

        aero = getattr(sol.data, f"modalaeroroger_{sys.name}")
        num_poles = sys.aero.num_poles

        return (gamma1, gamma2, omega, phi1l, psi2l, force_gravity,
                states, X_xdelta, C0ab, component_names, num_nodes,
                component_nodes, component_father, num_modes, num_poles,
                aero.A0hat, aero.A1hat, aero.A2hatinv, aero.A3hat,
                u_inf, c_ref, aero.poles)

@catter2library
def arg_20G21(sol: solution.IntrinsicSolution,
              sys: intrinsicmodal.Dsystem,
              fem: intrinsicmodal.Dfem,
              *args, **kwargs) -> tuple:
    
        phi1l = sol.data.modes.phi1l
        psi2l = sol.data.modes.psi2l
        gamma1 = sol.data.couplings.gamma1
        gamma2 = sol.data.couplings.gamma2
        omega = sol.data.modes.omega
        states = sys.states
        u_inf = sys.aero.u_inf
        c_ref = sys.aero.c_ref
        num_modes = fem.num_modes
        force_gravity = sys.xloads.force_gravity
        X_xdelta = sol.data.modes.X_xdelta
        C0ab = sol.data.modes.C0ab

        num_nodes = fem.num_nodes
        component_nodes = fem.component_nodes_int
        component_names = fem.component_names_int
        component_father = fem.component_father_int

        aero = getattr(sol.data, f"modalaeroroger_{sys.name}")
        num_poles = sys.aero.num_poles

#     """Clamped structure with gravity and Roger aero/gust"""
#     (gamma1, gamma2, omega, phi1l, psi2l, force_gravity,
#     states, X_xdelta, C0ab, component_names, num_nodes,
#      component_nodes, component_father, num_modes, num_poles,
#      A0hat, A1hat, A2hatinv, A3hat,
#      u_inf, c_ref, poles,
#      xgust, Flgust, F1gust) = args[0]

@catter2library
def arg_20G27(sol: solution.IntrinsicSolution,
                sys: intrinsicmodal.Dsystem,
                fem: intrinsicmodal.Dfem,
                *args, **kwargs):
              
        phi1l = sol.data.modes.phi1l
        psi2l = sol.data.modes.psi2l
        gamma1 = sol.data.couplings.gamma1
        gamma2 = sol.data.couplings.gamma2
        omega = sol.data.modes.omega
        states = sys.states
        u_inf = sys.aero.u_inf
        c_ref = sys.aero.c_ref
        num_modes = fem.num_modes
        force_gravity = sys.xloads.force_gravity
        X_xdelta = sol.data.modes.X_xdelta
        C0ab = sol.data.modes.C0ab

        num_nodes = fem.num_nodes
        component_nodes = fem.component_nodes_int
        component_names = fem.component_names_int
        component_father = fem.component_father_int

        aero = getattr(sol.data, f"modalaerostatespace_{sys.name}")

        return(gamma1, gamma2, omega, phi1l, psi2l, force_gravity, 
                states, X_xdelta, C0ab, component_names, num_nodes, 
                component_nodes, component_father,
                aero.Ahat, aero.B0hat, aero.B1hat, aero.Chat, aero.D0hat, aero.D1hat)

############################################
@catter2library
def arg_001001(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               fem: intrinsicmodal.Dfem,
               t: float,
               *args, **kwargs):

    gamma2 = sol.data.couplings.gamma2
    phi1 = sol.data.modes.phi1l
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower
    return (gamma2, omega, phi1, x,
            force_follower, t)


@catter2library
def arg_0011(sol: solution.IntrinsicSolution,
             system: intrinsicmodal.Dsystem,
             fem: intrinsicmodal.Dfem,
             t: float,
             *args, **kwargs):

    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    A0 = sol.data.modalaeroroger.A0
    C0 = sol.data.modalaeroroger.C0
    qalpha = system.aero.qalpha
    u_inf = system.aero.u_inf
    rho_inf = system.aero.rho_inf
    return (gamma2, omega,
            u_inf, rho_inf,
            qalpha, A0, C0)

@catter2library
def arg_000001(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               fem: intrinsicmodal.Dfem,
               t: float,
               *args, **kwargs):

    phi1 = sol.data.modes.phi1l
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower
    return (omega, phi1, x,
            force_follower, t)

@catter2library
def arg_00101(sol: solution.IntrinsicSolution,
              system: intrinsicmodal.Dsystem,
              fem: intrinsicmodal.Dfem,
              t: float,
              *args, **kwargs):

    phi1l = sol.data.modes.phi1l
    psi2l = sol.data.modes.psi2l 
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_dead = system.xloads.force_dead
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma2, omega, phi1l, psi2l,
            x, force_dead,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father, t)

@catter2library
def arg_101000(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               *args, **kwargs):

    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    states = system.states
    return gamma1, gamma2, omega, states

@catter2library
def arg_101001(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               *args, **kwargs):

    phi1 = sol.data.modes.phi1l    
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower    
    states = system.states
    return (gamma1, gamma2, omega, phi1,
            x, force_follower, states)

@catter2library
def arg_100001(sol: solution.IntrinsicSolution,
               system: intrinsicmodal.Dsystem,
               *args, **kwargs):

    phi1 = sol.data.modes.phi1l    
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_follower = system.xloads.force_follower    
    states = system.states
    return (omega, phi1,
            x, force_follower, states)

@catter2library
def arg_10101(sol: solution.IntrinsicSolution,
              system: intrinsicmodal.Dsystem,
              fem: intrinsicmodal.Dfem,
              *args, **kwargs):

    phi1 = sol.data.modes.phi1l
    psi2 = sol.data.modes.psi2l 
    gamma1 = sol.data.couplings.gamma1
    gamma2 = sol.data.couplings.gamma2
    omega = sol.data.modes.omega
    x = system.xloads.x
    force_dead = system.xloads.force_dead
    states = system.states
    X_xdelta = sol.data.modes.X_xdelta
    C0ab = sol.data.modes.C0ab
    num_nodes = fem.num_nodes
    component_nodes = fem.component_nodes_int
    component_names = fem.component_names_int
    component_father = fem.component_father_int
    return (gamma1, gamma2, omega, phi1, psi2,
            x, force_dead, states,
            X_xdelta,
            C0ab,
            component_names, num_nodes,
            component_nodes, component_father)