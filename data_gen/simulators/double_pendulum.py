import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .base_simulator import BaseSimulator


class Simulator(BaseSimulator):

  def __init__(self, config):
    super(Simulator, self).__init__(config)

    constants = config.constants
    self.g = self.load_constant(constants.g, 'g')
    self.m1 = self.load_constant(constants.m1, 'm1')
    self.m2 = self.load_constant(constants.m2, 'm2')
    self.l1 = self.load_constant(constants.l1, 'l1')
    self.l2 = self.load_constant(constants.l2, 'l2')
  
  def init_state(self):
    q0 = np.array([np.pi / 2, np.pi / 2 + 0])
    dq0 = np.array([0.0, 0.0])
    return q0, dq0

  def dynamics(self, t, y):
    q1, q2, dq1, dq2 = y
    a1 = (self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * np.cos(q1 - q2)
    a2 = (self.l1 / self.l2) * np.cos(q1 - q2)
    f1 = (
      - (self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * (dq2**2) * np.sin(q1 - q2)
      - (self.g / self.l1) * np.sin(q1)
    )
    f2 = (self.l1 / self.l2) * (dq1**2) * np.sin(q1 - q2) - (self.g / self.l2) * np.sin(q2)
    g1 = (f1 - a1 * f2) / (1 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1 - a1 * a2)
    ddq1 = g1
    ddq2 = g2
    return np.array([dq1, dq2, ddq1, ddq2])
  
  def lagrangian(self, q, dq):
    q1, q2 = q[:, 0], q[:, 1]
    dq1, dq2 = dq[:, 0], dq[:, 1]
    # kinetic energy (T)
    T1 = 0.5 * self.m1 * (self.l1 * dq1)**2
    T2 = 0.5 * self.m2 * (
      (self.l1 * dq1)**2 + (self.l2 * dq2)**2 +
      2 * self.l1 * self.l2 * dq1 * dq2 * np.cos(q1 - q2)
    )
    T = T1 + T2

    # potential energy (V)
    y1 = -self.l1 * np.cos(q1)
    y2 = y1 - self.l2 * np.cos(q2)
    V = self.m1 * self.g * y1 + self.m2 * self.g * y2
    return T - V
  
  def lagrangian_grad_q(self, q, dq):
    q1, q2 = q[:, 0], q[:, 1]
    dq1, dq2 = dq[:, 0], dq[:, 1]
    grad_q1 = (
      - self.m2 * self.l1 * self.l2 * dq1 * dq2 * np.sin(q1 - q2)
      - (self.m1 + self.m2) * self.g * self.l1 * np.sin(q1)
    )
    grad_q2 = (
      self.m2 * self.l1 * self.l2 * dq1 * dq2 * np.sin(q1 - q2)
      - self.m2 * self.g * self.l2 * np.sin(q2)
    )
    return np.stack([grad_q1, grad_q2], axis=-1)

  def lagrangian_grad_dq(self, q, dq):
    q1, q2 = q[:, 0], q[:, 1]
    dq1, dq2 = dq[:, 0], dq[:, 1]
    grad_dq1 = (
      (self.m1 + self.m2) * self.l1**2 * dq1 +
      self.m2 * self.l1 * self.l2 * dq2 * np.cos(q1 - q2)
    )
    grad_dq2 = (
      self.m2 * self.l2**2 * dq2 +
      self.m2 * self.l1 * self.l2 * dq1 * np.cos(q1 - q2)
    )
    return np.stack([grad_dq1, grad_dq2], axis=-1)
  
  def sample_trajectory(self):
    q0, dq0 = self.init_state()

    # Initial conditions: [initial angle, initial angular velocity]
    y0 = np.concatenate([q0, dq0], axis=0)
            
    # Solve the differential equation for this length
    sol = solve_ivp(self.dynamics, self.t_span, y0, dense_output=True, max_step=0.01)
    
    # Get time points and solutions (q, dq)
    t_vals = np.linspace(self.t_span[0], self.t_span[1], self.num_timesteps)
    y_vals = sol.sol(t_vals)
    q_vals, dq_vals = y_vals[:2].transpose((1, 0)), y_vals[2:].transpose((1, 0))
    
    # Compute Lagrangian at each time step
    L_vals = self.lagrangian(q_vals, dq_vals)
    L_grad_q = self.lagrangian_grad_q(q_vals, dq_vals)
    L_grad_dq = self.lagrangian_grad_dq(q_vals, dq_vals)
    
    # Save the trajectory, length, and Lagrangian
    trajectory = {
      'cond_dict': self.cond_dict,
      'time': t_vals,
      'q': q_vals,
      'dq': dq_vals,
      'L': L_vals,
      'L_grad_q': L_grad_q,
      'L_grad_dq': L_grad_dq,
    }
   
    return trajectory
  
  def visualize(self, traj, i_data):
    os.makedirs(self.vis_dir, exist_ok=True)

    ##################################################
    # Plot t-q curve
    ##################################################
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(traj['time'], traj['q'][:, 0], label=f'q1', color=self.color_palette[1])
    plt.plot(traj['time'], traj['q'][:, 1], label=f'q2', color=self.color_palette[4])
    plt.title(f'Angular displacement vs Time, l1={self.l1:.2f} m, l2={self.l2:.2f} m')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle [rad]')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(self.vis_dir, f'q-t_{i_data}.png')
    plt.savefig(save_path)

    ##################################################
    # Plot q1-dq1 curve
    ##################################################
    plt.figure(figsize=(10, 6))
    plt.plot(traj['q'][:, 0], traj['dq'][:, 0], label='q1', color=self.color_palette[1])
    plt.plot(traj['q'][:, 1], traj['dq'][:, 1], label='q2', color=self.color_palette[4])
    plt.title(f'Angular velocity vs Angular displacement, l1={self.l1:.2f} m, l2={self.l2:.2f} m')
    plt.xlabel('Angle [rad]')
    plt.ylabel('Angular velocity [rad/s]')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(self.vis_dir, f'dq-q_{i_data}.png')
    plt.savefig(save_path)

    ##################################################
    # Plot video
    ##################################################

    t_per_frame = 5

    x1 = self.l1 * np.sin(traj['q'][:, 0])
    y1 = - self.l1 * np.cos(traj['q'][:, 0])
    x2 = x1 + self.l2 * np.sin(traj['q'][:, 1])
    y2 = y1 - self.l2 * np.cos(traj['q'][:, 1])

    fig, ax = plt.subplots()
    l_max = self.l1 + self.l2
    ax.set_xlim(-l_max-0.1, l_max+0.1)
    ax.set_ylim(-l_max-0.1, 0.5)
    ax.set_aspect('equal')
    title = ax.text(0.5, 1.05, '', ha='center', va='center', transform=ax.transAxes)

    ax.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Pendulum line and bob
    trace, = ax.plot([], [], '-', lw=2, alpha=0.7, color=self.color_palette[2])
    line, = ax.plot([], [], 'o-', lw=2, color=self.color_palette[0])

    # Trace history for the second pendulum bob
    trace_x, trace_y = [], []   
    
    def update(frame):
      t = frame * t_per_frame
      line.set_data([0, x1[t], x2[t]], [0, y1[t], y2[t]])

      # Append the trace of the second bob
      trace_x.append(x2[t])
      trace_y.append(y2[t])
      trace.set_data(trace_x, trace_y)

      title.set_text(f'Time step: {t:04d}')

      return line, trace, title

    # Create the animation
    ani = FuncAnimation(fig, update, frames=self.num_timesteps // t_per_frame, blit=True)
    save_path = os.path.join(self.vis_dir, f'video_{i_data}.gif')
    ani.save(save_path, writer='pillow', fps=15)