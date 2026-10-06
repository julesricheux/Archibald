"""
Validation : l'équilibre est fermé ICI (IPOPT via CasADi), jamais dans les méthodes.
Cas : Shoemaker Model 29 (Savitsky & Brown / Alourdas) b=16in, beta=20°, W=80 lb.
"""
import sys; sys.path.insert(0, '.')
import casadi as ca
from pkg import savitsky as sv, strip_theory as st
from pkg.registry import PLANING_RESISTANCE_METHODS as PR, PLANING_WRENCH_METHODS as PW

inch, lb = 0.0254, 4.4482216
b, beta, W, rho, nu, g = 16 * inch, 20., 80 * lb, 1000., 1.14e-6, 9.81

def solve(build, x0, lbx, ubx):
    x = ca.MX.sym('x', 2)
    F, M = build(x)
    nlp = dict(x=x, f=0, g=ca.vertcat(F, M))
    S = ca.nlpsol('s', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-10})
    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)
    return r['x'].full().ravel()

# --- A : Savitsky, états (h_tr = immersion de la quille au tableau, trim en DEGRES)
def build_sav(kt, lcg):
    def f(x):
        h, th = x[0], x[1]
        Lk = h / ca.tan(th * ca.pi / 180)         # hydrostatique : longueur de quille mouillée
        kw = dict(stw=kt, rho=rho, g=g, nu=nu, trim=th, deadrise=beta, Lwl=Lk, Bwl=b, lcg_tr=lcg)
        Fz = PW['savitsky']['Fz'](**kw); My = PW['savitsky']['My'](**kw)
        return (Fz - W) / W, My / (W * b)
    return f

for kt, lcg, ref in [(18.01, 18.5 * inch, (4.21, 15.12)), (20.92, 18.3 * inch, (3.46, 16.76))]:
    sol = solve(build_sav(kt, lcg), [0.06, 4.0], [0.001, 0.5], [0.5, 15.])
    h, th = sol
    kw = dict(stw=kt, rho=rho, g=g, nu=nu, trim=th, deadrise=beta, Lwl=h / float(ca.tan(th * 3.14159265358979 / 180)), Bwl=b, lcg_tr=lcg)
    R = float(PR['savitsky']['Rf'](**kw) + PR['savitsky']['Rw'](**kw))
    print(f"Savitsky {kt} kt : trim {th:.2f}° (réf {ref[0]})  R {R/lb:.2f} lb (réf {ref[1]})")

# --- B : strip 2D+T, états (zeta, trim) ; flottabilité = celle du modèle (stand-in de l'hydrostatique)
sec = st.prismatic_sections(1.6, b, beta, n=40)
def build_strip(kt, lcg):
    def f(x):
        o = st.strip_wrench(kt, rho, g, nu, x[0], x[1], sec, lcg, 0.10)
        return (o['Fz_dyn'] + o['Fz_hs'] - W) / W, (o['My_dyn'] + o['My_hs']) / (W * 0.8)
    return f

for kt, lcg in [(18.01, 18.5 * inch), (20.92, 18.3 * inch)]:
    z, th = solve(build_strip(kt, lcg), [0.03, 4.0], [-0.5, -3.], [1.0, 20.])
    o = st.strip_wrench(kt, rho, g, nu, z, th, sec, lcg, 0.10)
    R = float(o['Rp_dyn'] + o['Rp_hs'] + o['Rf'])
    print(f"Strip    {kt} kt : trim {float(th):.2f}°  R {R/lb:.2f} lb")
