import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 300
})


#We start defining functions for the acceleration to solve for the discrete and continuous systems
def acc_disc(q, m, kappa, omega0, alpha=0.0, beta=0.0, g3=0.0, g4=0.0):
    """
    Compute d2q_n/dt2 for the chain of coupled oscillators.
    Parameters:
        q: oscillator displacements q_n(t)
        m: individual oscillator mass (m = rho * a)
        kappa: harmonic spring constant between neighbors (kappa = tau / a)
        omega0: local on-site natural frequency (omega0 = mu * c)
        alpha, beta: cubic and quartic anharmonic link coefficients
        g3, g4: cubic and quartic local on-site nonlinear coefficients
    """
    dq_right = np.roll(q, -1) - q   # q_{n+1} - q_n
    dq_left  = q - np.roll(q, 1)    # q_n - q_{n-1}

    f_lineal = kappa * (dq_right - dq_left) - m * (omega0**2) * q
    f_site   = -g3 * (q**2) - g4 * (q**3)
    f_link   = (
        alpha * kappa * (dq_right**2 - dq_left**2)
        + beta * kappa * (dq_right**3 - dq_left**3)
    )
    return (f_lineal + f_site + f_link) / m


def acc_field(phi, k2, c, mu, lam3=0.0, lam4=0.0):
    """Compute d2phi/dt2 based on Klein-Gordon using spectral Fourier methods"""
    d2phi_dx2 = np.real(np.fft.ifft(-k2 * np.fft.fft(phi)))
    return (c**2) * d2phi_dx2 - (mu**2) * (c**2) * phi - lam3 * (phi**2) - lam4 * (phi**3)


#Solution based on the solver Velocity Verlet. See the reference in the pdf
def verlet_step(u, v, dt, func_acc, *args):
    """Temporal step dt which conserves the phase space structure"""
    v_mean = v + 0.5 * dt * func_acc(u, *args)
    u_new  = u + dt * v_mean
    v_new  = v_mean + 0.5 * dt * func_acc(u_new, *args)
    return u_new, v_new


#Energy analysis and Klein-Gordon expansions
def energy_local_osc(q, v, m, kappa, omega0, alpha=0.0, beta=0.0, g3=0.0, g4=0.0):
    """Individual energy h_n(t) in each oscillator"""
    dq_right = np.roll(q, -1) - q
    dq_left  = q - np.roll(q, 1)

    e_cin   = 0.5 * m * (v**2)
    e_site  = 0.5 * m * (omega0**2) * (q**2) + (g3 / 3.0) * (q**3) + (g4 / 4.0) * (q**4)
    v_right = 0.5 * kappa * (dq_right**2) + (alpha * kappa / 3.0) * (dq_right**3) + (beta * kappa / 4.0) * (dq_right**4)
    v_left  = 0.5 * kappa * (dq_left**2) + (alpha * kappa / 3.0) * (dq_left**3) + (beta * kappa / 4.0) * (dq_left**4)
    return e_cin + e_site + 0.5 * (v_right + v_left)


def energy_fourier_mode(q, v, m, kappa, omega0, n_modes=4):
    """Mode energy from the first n_modes harmonics (normalized by Parseval theorem)"""
    N = len(q)
    Q_k = np.fft.fft(q) / N
    V_k = np.fft.fft(v) / N
    k_idx = np.arange(N)
    omega2_k = omega0**2 + (4.0 * kappa / m) * (np.sin(np.pi * k_idx / N) ** 2)

    return np.array([
        N * m * (np.abs(V_k[k]) ** 2 + omega2_k[k] * (np.abs(Q_k[k]) ** 2))
        for k in range(1, n_modes + 1)
    ])


def free_kg_sol(x, t, phi0, pi0, L, c, mu):
    """Analytical solution from Klein-Gordon by Fourier expansion"""
    N = len(x)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    omega_kg = c * np.sqrt(k**2 + mu**2)

    phi_hat0  = np.fft.fft(phi0)
    pi_hat0   = np.fft.fft(pi0)
    phi_hat_t = np.zeros_like(phi_hat0)

    mask = omega_kg > 0.0
    w = omega_kg[mask]
    phi_hat_t[mask] = phi_hat0[mask] * np.cos(w * t) + pi_hat0[mask] * (np.sin(w * t) / w)
    if np.any(~mask):
        phi_hat_t[~mask] = phi_hat0[~mask] + pi_hat0[~mask] * t

    return np.real(np.fft.ifft(phi_hat_t))


def gauss_pack(x, L, c, A=0.45, sigma=0.85, k0=4.0 * np.pi / 10.0):
    """Initial condition, wave packet"""
    x0 = 0.5 * L
    env = A * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))
    phi0 = env * np.cos(k0 * (x - x0))
    dphi0 = env * (-(x - x0) / (sigma**2) * np.cos(k0 * (x - x0)) - k0 * np.sin(k0 * (x - x0)))
    pi0 = -c * dphi0
    return phi0, pi0


#Obtention of results
def simulation():
    #Macroscopic Constants
    L = 10.0
    rho, tau, mu = 1.0, 1.0, 0.8
    c = np.sqrt(tau / rho)
    lam3, lam4 = 0.25, 0.40
    alpha, beta = 0.35, 0.25
    nu_cfl = 0.02

    #Result 1: Discrete dynamics, energy modes and map
    N1 = 64
    a1 = L / N1
    m1, kappa1, omega0 = rho * a1, tau / a1, mu * c
    dt1 = nu_cfl * a1 / c
    T1 = 40.0
    steps1 = int(T1 / dt1)
    times1 = np.linspace(0, T1, steps1 + 1)

    #First excitation of the fundamental mode (stronger amplitude to observe modal transfer)
    n_idx = np.arange(N1)
    q = 2.5 * np.sin(2.0 * np.pi * n_idx / N1)
    v = np.zeros(N1)
    alpha_mod, beta_mod, omega0_mod = 1.2, 1.0, 0.2 * c

    E_modes = np.zeros((steps1 + 1, 4))
    H_total = np.zeros(steps1 + 1)
    E_modes[0] = energy_fourier_mode(q, v, m1, kappa1, omega0_mod, 4)
    H_total[0] = np.sum(energy_local_osc(q, v, m1, kappa1, omega0_mod, alpha_mod, beta_mod))

    for s in range(steps1):
        q, v = verlet_step(q, v, dt1, acc_disc, m1, kappa1, omega0_mod, alpha_mod, beta_mod)
        E_modes[s + 1] = energy_fourier_mode(q, v, m1, kappa1, omega0_mod, 4)
        H_total[s + 1] = np.sum(energy_local_osc(q, v, m1, kappa1, omega0_mod, alpha_mod, beta_mod))

    #Map of the energy for a traveling pulse
    x1 = n_idx * a1
    phi0_p, pi0_p = gauss_pack(x1, L, c, sigma=0.75)
    qp, vp = phi0_p / np.sqrt(rho), pi0_p / np.sqrt(rho)
    T_map = 12.0
    steps_map = int(T_map / dt1)
    jump = max(1, steps_map // 150)
    map_h = []
    t_map = []

    for s in range(steps_map + 1):
        if s % jump == 0:
            map_h.append(energy_local_osc(qp, vp, m1, kappa1, omega0, alpha, beta))
            t_map.append(s * dt1)
        qp, vp = verlet_step(qp, vp, dt1, acc_disc, m1, kappa1, omega0, alpha, beta)

    #Plot 1: Space-time energy map in a separate PDF
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.6))
    im = ax1.imshow(
        np.array(map_h), aspect="auto", origin="lower",
        extent=[0, N1 - 1, 0, t_map[-1]], cmap="inferno"
    )
    plt.colorbar(im, ax=ax1, label=r"Energía local $h_n(t)$")
    ax1.set_xlabel(r"Índice del oscilador $n$")
    ax1.set_ylabel(r"Tiempo $t$")
    ax1.set_title("Propagación espacio-temporal de energía")
    plt.tight_layout()
    plt.savefig("fig1_mapa_energia.pdf")
    plt.close(fig1)

    #Plot 2: Modal energy transfer in a separate PDF
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.6))
    for k in range(4):
        ax2.plot(times1, E_modes[:, k], label=rf"Modo $k={k+1}$")
    ax2.plot(times1, H_total, "k--", lw=1.2, label=r"Energía total $H(t)$")
    ax2.set_xlabel(r"Tiempo $t$")
    ax2.set_ylabel(r"Energía $E_k(t)$")
    ax2.set_title(r"Transferencia modal desde $k=1$")
    ax2.legend(loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig2_transferencia_modal.pdf")
    plt.close(fig2)

    #Result 2: Dynamics of the wave across multiple snapshots (oscillators vs Klein-Gordon)
    N_ref = 1024
    x_ref = np.arange(N_ref) * (L / N_ref)
    k2_ref = (2.0 * np.pi * np.fft.fftfreq(N_ref, d=L / N_ref)) ** 2
    phi_c, pi_c = gauss_pack(x_ref, L, c)
    dt_ref = nu_cfl * (L / N_ref) / c

    def init_lattice(N):
        a = L / N
        x_n = np.arange(N) * a
        m, kappa = rho * a, tau / a
        g3, g4 = a * (rho**1.5) * lam3, a * (rho**2.0) * lam4
        phi0, pi0 = gauss_pack(x_n, L, c)
        q_n, v_n = phi0 / np.sqrt(rho), pi0 / np.sqrt(rho)
        dt_n = nu_cfl * a / c
        return x_n, q_n, v_n, m, kappa, g3, g4, dt_n

    x32, q32, v32, m32, k32, g3_32, g4_32, dt32 = init_lattice(32)
    x128, q128, v128, m128, k128, g3_128, g4_128, dt128 = init_lattice(128)

    snapshots = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]
    fig3, axes = plt.subplots(2, 3, figsize=(10.8, 5.6), sharex=True, sharey=True)
    t_curr = 0.0

    for idx, T_snap in enumerate(snapshots):
        dt_interval = T_snap - t_curr
        if dt_interval > 0.0:
            steps_c = int(np.round(dt_interval / dt_ref))
            dt_c = dt_interval / steps_c
            for _ in range(steps_c):
                phi_c, pi_c = verlet_step(phi_c, pi_c, dt_c, acc_field, k2_ref, c, mu, lam3, lam4)

            steps_32 = int(np.round(dt_interval / dt32))
            dt_32 = dt_interval / steps_32
            for _ in range(steps_32):
                q32, v32 = verlet_step(q32, v32, dt_32, acc_disc, m32, k32, omega0, alpha, beta, g3_32, g4_32)

            steps_128 = int(np.round(dt_interval / dt128))
            dt_128 = dt_interval / steps_128
            for _ in range(steps_128):
                q128, v128 = verlet_step(q128, v128, dt_128, acc_disc, m128, k128, omega0, alpha, beta, g3_128, g4_128)

            t_curr = T_snap

        ax = axes.flat[idx]
        ax.plot(x_ref, phi_c, "k-", lw=1.4, label=r"Campo Klein-Gordon $\phi(x,t)$")
        ax.plot(x32, np.sqrt(rho) * q32, "bo--", ms=2.0, lw=0.7, alpha=0.75, label=r"Red $N=32$")
        ax.plot(x128, np.sqrt(rho) * q128, "ro", ms=1.8, alpha=0.85, label=r"Red $N=128$")
        ax.set_title(rf"$t = {T_snap:.1f}$")
        ax.set_ylim(-0.55, 0.55)
        ax.grid(True, alpha=0.3)

        if idx >= 3:
            ax.set_xlabel(r"Posición $x$")
        if idx % 3 == 0:
            ax.set_ylabel(r"Amplitud $\phi(x,t)$")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7.8)

    plt.tight_layout()
    plt.savefig("fig3_propagacion_snapshots.pdf")
    plt.close(fig3)

    #Result 3: Method comparison based on epsilon which quantifies the difference
    T_final = 4.0
    phi_field_T, pi_field_T = gauss_pack(x_ref, L, c)
    steps_c = int(np.round(T_final / dt_ref))
    dt_c = T_final / steps_c
    for _ in range(steps_c):
        phi_field_T, pi_field_T = verlet_step(phi_field_T, pi_field_T, dt_c, acc_field, k2_ref, c, mu, lam3, lam4)

    phi_c0, pi_c0 = gauss_pack(x_ref, L, c)
    phi_free_field_T = free_kg_sol(x_ref, T_final, phi_c0, pi_c0, L, c, mu)

    N_list = [32, 64, 128, 256, 512]
    a_vals, eps_free, eps_nl = [], [], []

    for N in N_list:
        a = L / N
        x_n = np.arange(N) * a
        m, kappa = rho * a, tau / a
        g3, g4 = a * (rho**1.5) * lam3, a * (rho**2.0) * lam4
        phi0, pi0 = gauss_pack(x_n, L, c)

        ql, vl   = phi0 / np.sqrt(rho), pi0 / np.sqrt(rho)
        qnl, vnl = phi0 / np.sqrt(rho), pi0 / np.sqrt(rho)

        dt = nu_cfl * a / c
        steps = int(np.round(T_final / dt))
        dt = T_final / steps

        for _ in range(steps):
            ql, vl   = verlet_step(ql, vl, dt, acc_disc, m, kappa, omega0)
            qnl, vnl = verlet_step(qnl, vnl, dt, acc_disc, m, kappa, omega0, 0.0, beta, g3, g4)

        grid_jump = N_ref // N
        err_l  = np.sqrt(a * np.sum((np.sqrt(rho) * ql - phi_free_field_T[::grid_jump]) ** 2))
        err_nl = np.sqrt(a * np.sum((np.sqrt(rho) * qnl - phi_field_T[::grid_jump]) ** 2))

        a_vals.append(a)
        eps_free.append(err_l)
        eps_nl.append(err_nl)

    a_vals = np.array(a_vals)
    eps_free = np.array(eps_free)
    eps_nl = np.array(eps_nl)
    pend_l  = np.polyfit(np.log(a_vals), np.log(eps_free), 1)[0]
    pend_nl = np.polyfit(np.log(a_vals), np.log(eps_nl), 1)[0]

    fig4, ax4 = plt.subplots(figsize=(5.6, 3.6))
    ax4.loglog(a_vals, eps_free, "o-", label=rf"Régimen libre (pendiente = {pend_l:.2f})")
    ax4.loglog(a_vals, eps_nl, "s--", label=rf"Interactuante (pendiente = {pend_nl:.2f})")
    ax4.loglog(a_vals, eps_nl[-1] * (a_vals / a_vals[-1]) ** 2, "k:", label=r"Referencia $\mathcal{O}(a^2)$")
    ax4.set_xlabel(r"Espaciado de red $a = \ell/N$")
    ax4.set_ylabel(r"Error $\epsilon_{L^2}(a, T)$")
    ax4.set_title(r"Convergencia discreto $\to$ continuo al reducir $a$")
    ax4.legend()
    ax4.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig4_error_epsilon.pdf")
    plt.close(fig4)


#Animation of the dynamics
def animation_prop(N_disc=64):
    """Animation of how the oscillators move"""
    L, rho, tau, mu = 10.0, 1.0, 1.0, 0.8
    c = np.sqrt(tau / rho)
    lam3, lam4, alpha, beta = 0.25, 0.40, 0.35, 0.25

    N_cont = 512
    x_c = np.arange(N_cont) * (L / N_cont)
    k2_c = (2.0 * np.pi * np.fft.fftfreq(N_cont, d=L / N_cont)) ** 2
    phi_c, pi_c = gauss_pack(x_c, L, c)

    a = L / N_disc
    x_d = np.arange(N_disc) * a
    m, kappa, omega0 = rho * a, tau / a, mu * c
    g3, g4 = a * (rho**1.5) * lam3, a * (rho**2.0) * lam4
    phi0_d, pi0_d = gauss_pack(x_d, L, c)
    q_d, v_d = phi0_d / np.sqrt(rho), pi0_d / np.sqrt(rho)

    dt = 0.02 * a / c
    substeps = 10  #Each frame is 10 steps

    fig, ax = plt.subplots(figsize=(7, 3.8))
    line_cont, = ax.plot(x_c, phi_c, "k-", lw=1.5, label="Campo continuo Klein-Gordon")
    points_disc, = ax.plot(x_d, np.sqrt(rho) * q_d, "ro", ms=2.2, alpha=0.9, label=f"Osciladores (N={N_disc})")
    text_t = ax.text(0.02, 0.90, "t = 0.00", transform=ax.transAxes)
    ax.set_xlim(0, L)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Posición espacial x")
    ax.set_ylabel("Amplitud")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    state = {"q": q_d, "v": v_d, "phi": phi_c, "pi": pi_c, "t": 0.0}

    def actualize(_frame):
        for _ in range(substeps):
            state["q"], state["v"] = verlet_step(state["q"], state["v"], dt, acc_disc, m, kappa, omega0, alpha, beta, g3, g4)
            state["phi"], state["pi"] = verlet_step(state["phi"], state["pi"], dt, acc_field, k2_c, c, mu, lam3, lam4)
            state["t"] += dt
        line_cont.set_ydata(state["phi"])
        points_disc.set_ydata(np.sqrt(rho) * state["q"])
        text_t.set_text(f"t = {state['t']:.2f}")
        return line_cont, points_disc, text_t

    ani = FuncAnimation(fig, actualize, frames=200, interval=30, blit=True)

    #Save the animation video to disk before showing the window
    try:
        ani.save("animacion_propagacion.mp4", writer="ffmpeg", fps=30, dpi=200)
    except Exception:
        ani.save("animacion_propagacion.gif", writer="pillow", fps=30)

    plt.show()
    return ani


if __name__ == "__main__":
    simulation()
    animation_prop(N_disc=64)
