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
    self.m = self.load_constant(constants.m, 'm')
    self.l = self.load_constant(constants.l, 'l')
  
  def init_state(self):
    q0 = np.pi / 2
    dq0 = 0.0
    return q0, dq0

  def dynamics(self, t, y):
    q, dq = y
    ddq = -(self.g / self.l) * np.sin(q)
    return [dq, ddq]
  
  def lagrangian(self, q, dq):
    T = 0.5 * self.m * (self.l ** 2) * (dq ** 2)    # Kinetic energy
    V = self.m * self.g * self.l * (1 - np.cos(q))    # Potential energy
    return T - V
  
  def lagrangian_grad_q(self, q, dq):
    """Gradient of L with respect to q."""
    return -self.m * self.g * self.l * np.sin(q)

  def lagrangian_grad_dq(self, q, dq):
    """Gradient of L with respect to dq."""
    return self.m * (self.l ** 2) * dq
  
  def sample_trajectory(self):
    q0, dq0 = self.init_state()

    # Initial conditions: [initial angle, initial angular velocity]
    y0 = [q0, dq0]
            
    # Solve the differential equation for this length
    sol = solve_ivp(self.dynamics, self.t_span, y0, dense_output=True, max_step=0.01)
    
    # Get time points and solutions (q, dq)
    t_vals = np.linspace(self.t_span[0], self.t_span[1], self.num_timesteps)
    y_vals = sol.sol(t_vals)
    q_vals, dq_vals = y_vals
    q_vals, dq_vals = q_vals[:, None], dq_vals[:, None]
    
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
    plt.plot(traj['time'], traj['q'][:, 0], label=f'q1', color=self.color_palette[4])
    plt.title(f'Angular displacement vs Time, l={self.l:.2f} m')
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
    plt.plot(traj['q'][:, 0], traj['dq'][:, 0], label='q1', color=self.color_palette[4])
    plt.title(f'Angular velocity vs Angular displacement, l={self.l:.2f} m')
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

    x = self.l * np.sin(traj['q'][:, 0])
    y = - self.l * np.cos(traj['q'][:, 0])

    fig, ax = plt.subplots()
    l_max = self.l
    ax.set_xlim(-l_max-0.1, l_max+0.1)
    ax.set_ylim(-l_max-0.1, 0.1)
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
      line.set_data([0, x[t]], [0, y[t]])

      # Append the trace of the second bob
      trace_x.append(x[t])
      trace_y.append(y[t])
      trace.set_data(trace_x, trace_y)

      title.set_text(f'Time step: {t:04d}')

      return line, trace, title

    # Create the animation
    ani = FuncAnimation(fig, update, frames=self.num_timesteps // t_per_frame, blit=True)
    save_path = os.path.join(self.vis_dir, f'video_{i_data}.gif')
    ani.save(save_path, writer='pillow', fps=15)