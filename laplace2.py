# laplace updated to manually calc gradient instead of being lazy

import streamlit as st  # Streamlit for interactive GUI
import numpy as np  
import matplotlib.pyplot as plt  # Static plotting
import plotly.graph_objects as go  # Interactive 3D plotting

# ------------------------------------------------------------------
# function: Gauss–Seidel iteration with Successive Over-Relaxation (SOR)
# ------------------------------------------------------------------
def solve_laplace_sor(phi, mask, tol, omega, max_iter, animate=False, interval=50):
    """
    Solve Laplace's equation using Gauss-Seidel with SOR.

    Parameters:
        phi (np.ndarray): Potential array (N x N).
        mask (np.ndarray): Boolean mask, True for fixed points.
        tol (float): Convergence tolerance.
        omega (float): SOR relaxation parameter (1 < ω < 2).
        max_iter (int): Maximum iterations.
        animate (bool): Whether to yield intermediate frames.
        interval (int): Interval between animation frames.

    Yields:
        it (int): Iteration number.
        phi (np.ndarray): Current potential.
        converged (bool): Whether solution has converged.
    """
    n, m = phi.shape  # grid dimensions
    for it in range(1, max_iter + 1):
        max_diff = 0.0  # track largest update this iteration
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                if not mask[i, j]:  # skip boundary/fixed points
                    new_val = 0.25 * (
                        phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1]
                    )
                    diff = new_val - phi[i, j]
                    phi[i, j] += omega * diff
                    max_diff = max(max_diff, abs(diff))
        if max_diff < tol:
            if animate:
                yield it, np.copy(phi), True
            break
        if animate and it % interval == 0:
            yield it, np.copy(phi), False
    else:
        it = max_iter
    yield it, np.copy(phi), (max_diff < tol)

# -----------------------------------
# function to compute electric field using manual finite differences
# -----------------------------------
def compute_field(phi, h):
    """
    Compute electric field from potential using finite differences.
    Returns (Ex, Ey).
    """
    dphi_dx = np.zeros_like(phi)
    dphi_dy = np.zeros_like(phi)
    # Central difference for interior points
    dphi_dx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * h)
    dphi_dy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * h)
    # One-sided differences for boundaries
    dphi_dx[:, 0] = (phi[:, 1] - phi[:, 0]) / h   # left edge
    dphi_dx[:, -1] = (phi[:, -1] - phi[:, -2]) / h # right edge
    dphi_dy[0, :] = (phi[1, :] - phi[0, :]) / h    # top edge
    dphi_dy[-1, :] = (phi[-1, :] - phi[-2, :]) / h # bottom edge
    Ex = dphi_dx
    Ey = dphi_dy
    return Ex, Ey

# -----------------------------------
# Streamlit application layout
# -----------------------------------
st.set_page_config(page_title="Laplace Equation Visualizer", layout="wide", page_icon="⚡")
st.title("Laplace Equation Visualizer")

with st.sidebar:
    with st.expander("About this app"):
        st.markdown("""
        **Laplace Equation Visualizer**

        This interactive app numerically solves the 2D Laplace equation  
        using the Gauss-Seidel method with Successive Over-Relaxation (SOR)  
        on a square grid with customizable boundary voltages (Dirichlet conditions).

        Features:
        - Adjustable grid size and boundary voltages
        - Animated convergence (optional)
        - 2D and 3D visualization of electric potential
        - Visualization of electric field (gradient)
        - 1D cross-sections for analysis

        The solution shows the steady-state electric potential inside the box.(hopefully)
        """)

    st.markdown("<hr style='border:1px solid #00BFFF'>", unsafe_allow_html=True)

    st.header("Grid & Solver Parameters")
    N = st.slider(
        "Grid Resolution (N × N)",
        min_value=20, max_value=300, value=100, step=10,
        help="Sets number of grid points in each direction (higher N = higher accuracy, slower solve)"
    )
    with st.expander("What is Grid Resolution?"):
        st.markdown(
            "Sets the number of points in each direction for the grid. "
            "A higher value increases accuracy, but also increases computation time. "
            "**Tip:** Try starting with 100 for balance."
        )

    # Compute grid spacing and minimum useful tolerance (discretization error)
    h = 1.0 / (N - 1)
    min_tol = h**2

    st.markdown(
        f"<span style='font-size: 0.9em'>"
        f"Grid spacing: <b>$h = \\frac{{1}}{{N-1}} = {h:.5f}$</b>"
        "</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border:1px solid #00BFFF'>", unsafe_allow_html=True)
    st.subheader("Boundary Voltages (V)")
    V_top = st.number_input("Top", value=1.00, step=0.50, format="%.2f", help="Potential on top edge")
    V_bottom = st.number_input("Bottom", value=0.00, step=0.50, format="%.2f", help="Potential on bottom edge")
    V_left = st.number_input("Left", value=0.00, step=0.50, format="%.2f", help="Potential on left edge")
    V_right = st.number_input("Right", value=0.00, step=0.50, format="%.2f", help="Potential on right edge")

    st.markdown("<hr style='border:1px solid #00BFFF'>", unsafe_allow_html=True)
    
    st.subheader("Solver Parameters")
    tol = st.number_input(
        "Tolerance (ε)", min_value=1e-12, max_value=1e-2, value=1e-4, format="%.1e",
        help="Smallest allowed change at any point before solver stops (convergence criterion)"
    )
    # Message for small tolerance
    if tol < 1e-10:
        st.info("That's a very small tolerance. Remember, grid accuracy limits it.")

    # Show recommended minimum tol
    st.markdown(
        f"<span style='font-size: 0.9em'>"
        f"Recommended minimum tol:<br><b>$h^2 = {min_tol:.2e}$</b>"
        "</span>",
        unsafe_allow_html=True,
    )
    with st.expander("What is Tolerance?"):
        st.markdown(
            "Sets the smallest change in the solution allowed at each step before the solver stops. "
            "Lower values make the result more precise, but take longer to compute. "
            "Don't set it smaller than the recommended $h^2$."
        )

    omega_opt = 2.0 / (1.0 + np.sin(np.pi / (N - 1)))
    omega = st.slider(
        "Relaxation $\\omega$", min_value=1.0, max_value=1.95, value=1.5, step=0.05,
        help="Controls speed of convergence (optimal ω is shown below)"
    )
    st.write(f"Optimal SOR $\\omega \\approx$ {omega_opt:.4f} for N = {N}")
    with st.expander("What is Relaxation $\\omega$?"):
        st.markdown(
            r"""
            Controls the speed of convergence for the Gauss-Seidel solver.
            - $\omega = 1$ means standard Gauss-Seidel (slower).
            - Values up to 2 can accelerate convergence.
            **Tip:** Use the suggested optimal value for your grid.
            """
        )

    max_iter = st.number_input(
        "Max Iterations", min_value=100, max_value=20000, value=5000, step=100,
        help="Maximum allowed solver iterations"
    )
    with st.expander("What is Max Iterations?"):
        st.markdown(
            "Sets the maximum number of update cycles before the solver gives up. "
            "If the solver hasn't converged by this point, it will stop anyway. "
            "**Tip:** Use higher values if you set a very low tolerance or use a large grid."
        )

    st.markdown("<hr style='border:1px solid #00BFFF'>", unsafe_allow_html=True)
    st.subheader("Display Options")
    # Colormap selector
    colormap = st.selectbox("Colormap", ["viridis", "plasma", "magma", "cividis", "turbo"], index=0)
    show_field  = st.checkbox("Electric Field (streamlines)", value=False)
    show_slices = st.checkbox("1D Slices", value=False)
    show_3d_static = st.checkbox("3D Surface (static)", value=False)
    show_3d_interact = st.checkbox("3D Surface (interactive)", value=False)
    animate_conv = st.checkbox("Animate Convergence", value=False)
    run_solver = st.button("Run Solver")

# ----- End sidebar -----

if run_solver:
    phi  = np.zeros((N, N), dtype=float)
    mask = np.zeros_like(phi, dtype=bool)
    phi[0,:] = V_top; mask[0,:] = True  # top edge
    phi[-1,:] = V_bottom; mask[-1, :] = True  # bottom edge
    phi[:,0] = V_left; mask[:, 0] = True  # left edge
    phi[:,-1] = V_right; mask[:, -1] = True  # right edge

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    if animate_conv:
        placeholder = st.empty()
        progress = st.progress(0)
        for it, phi_frame, converged in solve_laplace_sor(
            phi.copy(), mask, tol, omega, max_iter,
            animate=True, interval=50
        ):
            progress.progress(min(it / max_iter, 1.0))
            Ex, Ey = compute_field(phi_frame, h)
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(phi_frame, cmap=colormap, extent=[0, 1, 0, 1])
            ax.set_title(f"Iteration {it}")
            plt.colorbar(im, ax=ax, label="φ (V)")
            if show_field:
                ax.streamplot(X, Y, Ex, Ey,
                              density=1.2, color='white', linewidth=0.7,
                              arrowsize=1, arrowstyle='->')
            placeholder.pyplot(fig)
            plt.close(fig)
            if converged:
                break
        status = f"Converged in {it} iterations." if converged else "Did not converge."
        st.sidebar.success(status)
        phi = phi_frame.copy()
    else:
        it, phi, converged = next(
            solve_laplace_sor(phi.copy(), mask, tol, omega, max_iter)
        )
        msg = f"Converged in {it} iterations." if converged else "Did not converge."
        if converged:
            st.sidebar.success(msg)
        else:
            st.sidebar.warning(msg)

    # 2D Potential Plot
    st.subheader("2D Potential φ(x, y)")
    fig2d, ax2d = plt.subplots(figsize=(5, 4))
    im2d = ax2d.imshow(phi, cmap=colormap, extent=[0, 1, 0, 1])
    plt.colorbar(im2d, ax=ax2d, label="φ (V)")
    if show_field:
        Ex, Ey = compute_field(phi, h)
        ax2d.streamplot(X, Y, Ex, Ey,
                         density=1.2, color='white', linewidth=0.7,
                         arrowsize=1, arrowstyle='->')
    st.pyplot(fig2d)
    plt.close(fig2d)

    # 1D Slices Plot
    if show_slices:
        st.subheader("1D Slices of φ")
        mid = N // 2
        idx = np.arange(N)
        fig1d, (axh, axv) = plt.subplots(1, 2, figsize=(8, 3))
        axh.plot(idx, phi[mid, :])
        axh.set(title=f"φ at y={mid}", xlabel="x index", ylabel="φ (V)")
        axh.grid(True)
        axv.plot(idx, phi[:, mid])
        axv.set(title=f"φ at x={mid}", xlabel="y index", ylabel="φ (V)")
        axv.grid(True)
        st.pyplot(fig1d)
        plt.close(fig1d)

    # Static 3D Surface Plot
    if show_3d_static:
        st.subheader("3D Surface (static)")
        fig3d, ax3d = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(8, 6))
        surf = ax3d.plot_surface(X, Y, phi, cmap=colormap, edgecolor='none')
        
        # Set z-limits to min/max of phi (axis kept getting cut off)
        zmin = float(np.min(phi))
        zmax = float(np.max(phi))
        if np.isclose(zmax, zmin):
            zmin -= 0.1
            zmax += 0.1
        ax3d.set_zlim(zmin, zmax)
        
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("φ (V)")
        # Add colorbar
        fig3d.colorbar(surf, shrink=0.6, aspect=10, pad=0.15, label="φ (V)")
        
        st.pyplot(fig3d)
        plt.close(fig3d)

    # Interactive 3D Surface Plot
    if show_3d_interact:
        st.subheader("3D Surface (interactive)")
        fig_p = go.Figure(
            data=[go.Surface(
                x=X, y=Y, z=phi,
                colorscale=colormap.capitalize(),
                showscale=True,
                colorbar=dict(title="φ (V)")
            )]
        )
        fig_p.update_layout(
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="φ(V)",
                aspectmode='cube'
            ),
            width=700,
            height=600,
        )
        st.plotly_chart(fig_p, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.write("Done.")


