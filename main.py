import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# DH parameters: [alpha, a, d, theta_offset]
DH = np.array([
    [ np.pi/2,   0.0,   0.40,   0.0 ],
    [ 0.0,       0.35,  0.0,    0.0 ],
    [ 0.0,       0.30,  0.0,    0.0 ],
    [ np.pi/2,   0.0,   0.20,   0.0 ],
    [-np.pi/2,   0.0,   0.0,    0.0 ],
    [ 0.0,       0.0,   0.10,   0.0 ],
], dtype=float)

N = len(DH)

# Joint angle limits (radians)
JOINT_LIMITS = np.array([
    [-np.pi, np.pi],
    [-2*np.pi/3, 2*np.pi/3],
    [-2*np.pi/3, 2*np.pi/3],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
])

# DLS inverse kinematics parameters
LAMBDA = 0.05
ALPHA = 0.5


def dh_matrix(alpha, a, d, theta):
    # Compute one DH transformation matrix
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])


def forward_kinematics(q):
    # Compute transforms from base to every joint
    T = np.eye(4)
    Ts = [T.copy()]

    for i in range(N):
        alpha, a, d, off = DH[i]
        T = T @ dh_matrix(alpha, a, d, q[i] + off)
        Ts.append(T.copy())

    return Ts


def compute_jacobian(q):
    # Compute 6xN geometric Jacobian
    Ts = forward_kinematics(q)
    pe = Ts[-1][:3, 3]

    J = np.zeros((6, N))

    for i in range(N):
        zi = Ts[i][:3, 2]
        pi = Ts[i][:3, 3]

        J[:3, i] = np.cross(zi, pe - pi)
        J[3:, i] = zi

    return J


def dls_step(q, target):
    # One damped least squares IK update
    Ts = forward_kinematics(q)
    pe = Ts[-1][:3, 3]

    e = target - pe
    J = compute_jacobian(q)

    J_dls = J.T @ np.linalg.inv(J @ J.T + (LAMBDA**2) * np.eye(6))
    dq = ALPHA * (J_dls[:, :3] @ e)

    q_new = q + dq

    # Enforce joint limits
    return np.clip(q_new, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]), np.linalg.norm(e)


class RobotViz:
    # Interactive robot visualization and simulation
    def __init__(self):
        self.q = np.zeros(N)
        self.target = np.array([0.3, 0.3, 0.4])
        self.running = True

        self.fig = plt.figure(figsize=(14, 8))

        # 3D robot plot
        self.ax_plot = self.fig.add_axes(
            [0.05, 0.35, 0.45, 0.60],
            projection='3d'
        )

        # Control panel area
        self.ax_ctrl = self.fig.add_axes([0.05, 0.05, 0.45, 0.25])
        self.ax_ctrl.axis('off')

        # Text panel for matrices and error
        self.calc_text = self.fig.text(
            0.70, 0.95, "",
            fontsize=8,
            family="monospace",
            va='top',
            ha='left',
            bbox=dict(
                facecolor='white',
                alpha=0.9,
                edgecolor='black'
            )
        )

        self._setup_axes()
        self._build_controls()

    def _setup_axes(self):
        # Configure 3D axes
        ax = self.ax_plot
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 1.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    def _build_controls(self):
        # Create sliders and buttons
        axcolor = 'lightgray'

        def place(y):
            return self.fig.add_axes(
                [0.10, y, 0.30, 0.025],
                facecolor=axcolor
            )

        # Target position sliders
        self.sx = Slider(place(0.22), 'X', -0.8, 0.8,
                         valinit=self.target[0])
        self.sy = Slider(place(0.18), 'Y', -0.8, 0.8,
                         valinit=self.target[1])
        self.sz = Slider(place(0.14), 'Z', 0.0, 1.2,
                         valinit=self.target[2])

        # IK parameter sliders
        self.sd = Slider(place(0.10), 'Damping', 0.001, 0.3,
                         valinit=LAMBDA)
        self.sa = Slider(place(0.06), 'Step', 0.05, 1.5,
                         valinit=ALPHA)

        self.sx.on_changed(self.update_target)
        self.sy.on_changed(self.update_target)
        self.sz.on_changed(self.update_target)
        self.sd.on_changed(self.update_params)
        self.sa.on_changed(self.update_params)

        # Control buttons
        self.btn_reset = Button(
            self.fig.add_axes([0.48, 0.08, 0.06, 0.05]),
            "Reset"
        )
        self.btn_pause = Button(
            self.fig.add_axes([0.48, 0.15, 0.06, 0.05]),
            "Pause"
        )

        self.btn_reset.on_clicked(self.reset)
        self.btn_pause.on_clicked(self.toggle)

    def update_target(self, val):
        # Update target from sliders
        self.target = np.array([
            self.sx.val,
            self.sy.val,
            self.sz.val
        ])

    def update_params(self, val):
        # Update DLS parameters
        global LAMBDA, ALPHA
        LAMBDA = self.sd.val
        ALPHA = self.sa.val

    def reset(self, event):
        # Reset joint angles
        self.q[:] = 0

    def toggle(self, event):
        # Pause or resume simulation
        self.running = not self.running

    def format_matrix(self, M, name):
        # Convert matrix to formatted string
        s = f"{name}:\n"
        for row in M:
            s += "  [" + " ".join(f"{v:+.2f}" for v in row) + "]\n"
        return s + "\n"

    def build_calc_text(self):
        # Build information text panel
        Ts = forward_kinematics(self.q)
        J = compute_jacobian(self.q)
        pe = Ts[-1][:3, 3]

        text = "FORWARD KINEMATICS\n"

        for i in range(1, min(4, len(Ts))):
            text += self.format_matrix(Ts[i], f"T0{i}")

        text += f"EE POS:\n[{pe[0]:+.3f}, {pe[1]:+.3f}, {pe[2]:+.3f}]\n\n"

        text += "JACOBIAN:\n"
        text += self.format_matrix(J, "J")

        err = self.target - pe
        text += "ERROR:\n"
        text += f"[{err[0]:+.3f}, {err[1]:+.3f}, {err[2]:+.3f}]"

        return text

    def draw(self):
        # Draw robot and target
        ax = self.ax_plot
        ax.cla()
        self._setup_axes()

        Ts = forward_kinematics(self.q)
        pts = np.array([T[:3, 3] for T in Ts])

        # Draw links
        for i in range(N):
            ax.plot(
                [pts[i, 0], pts[i+1, 0]],
                [pts[i, 1], pts[i+1, 1]],
                [pts[i, 2], pts[i+1, 2]],
                linewidth=6,
                color='blue'
            )

        # Draw joints
        for p in pts:
            ax.scatter(*p, s=100, color='black')

        # End effector
        ee = pts[-1]
        ax.scatter(*ee, s=150, color='red')

        # Target point
        ax.scatter(
            *self.target,
            s=200,
            color='orange',
            marker='X'
        )

        # Error line
        ax.plot(
            [ee[0], self.target[0]],
            [ee[1], self.target[1]],
            [ee[2], self.target[2]],
            linestyle='--'
        )

        self.calc_text.set_text(self.build_calc_text())
        self.fig.canvas.draw_idle()

    def step(self):
        # Run one simulation step
        if self.running:
            self.q, _ = dls_step(self.q, self.target)
        self.draw()

    def run(self):
        # Start update loop
        timer = self.fig.canvas.new_timer(interval=30)
        timer.add_callback(self.step)
        timer.start()
        plt.show()


if __name__ == "__main__":
    RobotViz().run()