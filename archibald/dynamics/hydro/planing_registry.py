"""
Registre de méthodes de résistance (pré-planing / planing) et outils de composition.

Boucle ouverte : aucune méthode ne résout quoi que ce soit ; l'assiette (trim, enfoncement)
et les propriétés hydrostatiques arrivent par kwargs, IPOPT ferme l'équilibre.
Même contrat que DEFAULT_RESISTANCE_METHODS :  méthode -> {composante: fonction}
chaque fonction est appelée avec les kwargs communs (stw, rho, g, nu, volume, ...).

Ajouter à votre DEFAULT_RESISTANCE_METHODS :
    from .planing_registry import PLANING_RESISTANCE_METHODS, PLANING_WRENCH_METHODS
    DEFAULT_RESISTANCE_METHODS.update(PLANING_RESISTANCE_METHODS)

ATTENTION : Fz et My sont des efforts, pas des résistances. Ils vivent dans
PLANING_WRENCH_METHODS pour ne jamais entrer dans la somme `_total`.
"""
from . import savitsky, strip_theory, radojcic
from .planing_common import speed_ms, Fn_vol, smoothstep

# Composantes de RESISTANCE (à sommer) : Rf, Rw, Rtr, ...
PLANING_RESISTANCE_METHODS = {
    "savitsky": {
        "Rf": savitsky.compute_Rf_savitsky,
        "Rw": savitsky.compute_Rw_savitsky,
        "Rtr": savitsky.compute_Rtr_savitsky,
    },
    "strip2dt": {
        "Rf": strip_theory.compute_Rf_strip,
        "Rw": strip_theory.compute_Rw_strip,
        "Rtr": savitsky.compute_Rtr_savitsky,
    },
    # Active quand RADOJCIC_SIMPLE_COEFFS est renseigné :
    "radojcic": {
        "Rf": radojcic.compute_Rf_radojcic,
        "Rw": radojcic.compute_Rw_radojcic,
        "Rtr": savitsky.compute_Rtr_savitsky,
    },
}

# Torseur hydrodynamique (NE PAS sommer avec les résistances) : Fz [N], My [N.m]
PLANING_WRENCH_METHODS = {
    "savitsky": {
        "Fz": savitsky.compute_Fz_savitsky,
        "My": savitsky.compute_My_savitsky,
    },
    "strip2dt": {
        "Fz": strip_theory.compute_Fz_strip,
        "My": strip_theory.compute_My_strip,
    },
}

# Compatibilité : ancien nom = résistances uniquement
PLANING_METHODS = PLANING_RESISTANCE_METHODS


def compose(methods, **choices):
    """
    Assemble des composantes de méthodes différentes.
        compose(DEFAULT_RESISTANCE_METHODS, Rf="savitsky", Rw="savitsky", Rtr="savitsky")
    Attention : si Rf et Rw viennent de méthodes différentes, vérifier qu'ils
    représentent la même décomposition (Rf avec surface/longueur mouillées dynamiques).
    """
    return {comp: methods[name][comp] for comp, name in choices.items()}


def blend_methods(methods, low, high, x0, x1, key="Fn_vol"):
    """
    Méthode composite : (1-w) * low + w * high, w = smoothstep(Fn_vol, x0, x1).
    Exemple : blend_methods(M, "dsyhs", "savitsky", 1.8, 2.6).
    Les composantes présentes dans les deux méthodes sont mélangées ; les autres
    sont reprises telles quelles.
    """
    def make(fl, fh):
        def f(stw, rho, g, nu, volume, **kw):
            w = smoothstep(Fn_vol(speed_ms(stw), volume, g), x0, x1)
            return (1. - w) * fl(stw=stw, rho=rho, g=g, nu=nu, volume=volume, **kw) \
                + w * fh(stw=stw, rho=rho, g=g, nu=nu, volume=volume, **kw)
        return f
    out = {}
    for comp in set(methods[low]) | set(methods[high]):
        if comp in methods[low] and comp in methods[high]:
            out[comp] = make(methods[low][comp], methods[high][comp])
        else:
            out[comp] = methods[low].get(comp) or methods[high][comp]
    return out
