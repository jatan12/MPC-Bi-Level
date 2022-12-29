import sys
sys.path.append("./ddn") # Append the path to ddn directory

import warnings
warnings.filterwarnings('ignore')

# Generic imports
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from ddn.pytorch.node import LinEqConstDeclarativeNode

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float64) # No overflows with 64-bit precision

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prevents NaN by torch.log(0)
def torch_log(x):
	return torch.log(torch.clamp(x, min = 1e-10))

# Encoder Architecture
class Encoder(nn.Module):
	def __init__(self, inp_dim, out_dim, hidden_dim, z_dim):
		super(Encoder, self).__init__()
				
		# Encoder Architecture
		self.encoder = nn.Sequential(
			nn.Linear(inp_dim + out_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(), 
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU()
		)
		
		# Mean and Variance
		self.mu = nn.Linear(256, z_dim)
		self.var = nn.Linear(256, z_dim)

		# Softplus activation
		self.softplus = nn.Softplus()
		
	def forward(self, x):
		out = self.encoder(x)
		mu = self.mu(out)
		var = self.var(out)
		return mu, self.softplus(var)
	
# Decoder Architecture
class Decoder(nn.Module):
	def __init__(self, inp_dim, out_dim, hidden_dim, z_dim):
		super(Decoder, self).__init__()
		
		# Decoder Architecture
		self.decoder = nn.Sequential(
			nn.Linear(z_dim + inp_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			
			nn.Linear(256, out_dim),
		)
	
	def forward(self, x):
		out = self.decoder(x)
		return out

# Beta-cVAE Architecture with Differentiable Optimization Layer
class Beta_cVAE(nn.Module):
	def __init__(self, encoder, decoder, opt_layer, inp_mean, inp_std):
		super(Beta_cVAE, self).__init__()
		
		# Encoder & Decoder
		self.encoder = encoder
		self.decoder = decoder
		
		# Normalizing Constants
		self.inp_mean = inp_mean
		self.inp_std = inp_std
		
		# Differentiable Optimization Layer
		self.opt_layer = opt_layer

		# No. of Variables
		self.nvar = 11

		# Collision Cost Parameters
		t_fin = 15.0
		self.num = 100
		self.tot_time = torch.linspace(0, t_fin, self.num, device=device)
		self.num_obs = 10
		self.v_max = 30
		self.a_max = 8.0
		self.wheel_base = 2.5
		self.steer_max = 0.6
		self.v_des = 20.0
		self.k_p_v = 20.0
		
		# RCL Loss
		self.rcl_loss = nn.MSELoss()
			
	# Encoder: P_phi(z | x, y)
	def encode(self, x, y):
		inputs = torch.cat([x, y], dim = 1)
		mu, var = self.encoder(inputs)        
		return mu, var

	# Reparametrization Trick
	def reparameterize(self, mu, var):
		eps = torch.randn_like(mu, device=device)
		return mu + var * eps

	# Decoder: P_theta(y* | z, x) where y* is the optimal solution of the optimization problem
	def decode(self, z, x, init_state_ego):
		inputs = torch.cat([z, x], dim = 1)

		# Behavioral inputs
		y = self.decoder(inputs)

		# Minor Hack
		y[:, 0:4] = F.sigmoid(y[:, 0:4]) * 40. + 0.1 
		
		# Call Optimization Solver
		y_star = self.opt_layer(init_state_ego, y)

		return y_star

	# Collision Collision Cost
	def compute_col_cost(self, inp, y_star, traj_sol, Pdot, Pddot, y_ub, y_lb, a_obs=6.0, b_obs=3.2):
		
		num_batch = y_star.shape[0]

		# Obstacle coordinates & Velocity 
		x_obs = inp[:, 5::5]
		y_obs = inp[:, 6::5]
		vx_obs = inp[:, 7::5]
		vy_obs = inp[:, 8::5]

		# Coefficient Predictions        
		cx = y_star[:, 0:self.nvar]
		cy = y_star[:, self.nvar:2*self.nvar]

		# Trajectories               
		x = traj_sol[:, 0:self.num] 
		y = traj_sol[:, self.num:2*self.num]  

		xdot = torch.mm(Pdot, cx.T).T 
		ydot = torch.mm(Pdot, cy.T).T  

		xddot = torch.mm(Pddot, cx.T).T   
		yddot = torch.mm(Pddot, cy.T).T 

		# Batch Obstacle Trajectory Prediction
		x_obs_inp_trans = x_obs.reshape(num_batch, 1, self.num_obs)
		y_obs_inp_trans = y_obs.reshape(num_batch, 1, self.num_obs)

		vx_obs_inp_trans = vx_obs.reshape(num_batch, 1, self.num_obs)
		vy_obs_inp_trans = vy_obs.reshape(num_batch, 1, self.num_obs)

		x_obs_traj = x_obs_inp_trans + vx_obs_inp_trans * self.tot_time.unsqueeze(1)
		y_obs_traj = y_obs_inp_trans + vy_obs_inp_trans * self.tot_time.unsqueeze(1)

		x_obs_traj = x_obs_traj.permute(0, 2, 1)
		y_obs_traj = y_obs_traj.permute(0, 2, 1)

		x_obs_traj = x_obs_traj.reshape(num_batch, self.num_obs * self.num)
		y_obs_traj = y_obs_traj.reshape(num_batch, self.num_obs * self.num)

		x_extend = torch.tile(x, (1, self.num_obs))
		y_extend = torch.tile(y, (1, self.num_obs))

		wc_alpha = (x_extend - x_obs_traj)
		ws_alpha = (y_extend - y_obs_traj)
		dist_obs = - wc_alpha ** 2 / (a_obs ** 2) - ws_alpha ** 2 / (b_obs ** 2) + 1

		# Obstacle Penalty
		cost_obs_penalty = torch.sum(torch.maximum(torch.zeros((num_batch, self.num * self.num_obs), device=device), dist_obs), dim = 1) 
		
		# Steering Penalty
		curvature = (yddot * xdot - xddot * ydot) / ((xdot**2 + ydot**2) ** 1.5)
		steering = torch.arctan(curvature * self.wheel_base)
		steering_penalty = torch.sum(torch.maximum(torch.zeros((num_batch, self.num), device=device), torch.abs(steering) - self.steer_max), dim = 1)

		# Lane Penalty
		cost_y_penalty_ub = 2 * torch.sum(torch.maximum(torch.zeros((num_batch, self.num), device=device), y - y_ub.unsqueeze(1)), dim=1)
		cost_y_penalty_lb = 2 * torch.sum(torch.maximum(torch.zeros((num_batch, self.num), device=device), -y + y_lb.unsqueeze(1)), dim=1) 

		# V Desired Penalty
		temp_vx = torch.linalg.norm(xddot - self.k_p_v * (xdot - self.v_des), dim=1)

		# Total cost batch 
		cost_batch = torch.mean(cost_obs_penalty + steering_penalty + cost_y_penalty_ub + cost_y_penalty_lb + 0.015 * temp_vx) 
		
		return 0.5 * cost_batch
	
	# Loss Function
	def loss_function(self, mu, sigma, traj_sol, traj_gt, col_cost, beta = 1.0, step = 0, use_col=False):

		# Beta Annealing
		beta_d = min(step / 1000 * beta, beta)

		# KL Loss- 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KL = -0.5 * torch.mean(torch.sum(1 + torch_log(sigma ** 2) - mu ** 2 - sigma ** 2, dim=1))
		
		# RCL Loss 
		RCL = self.rcl_loss(traj_sol, traj_gt)  

		if not use_col:
			# ELBO
			loss = beta_d * KL + RCL
		else:					
			# ELBO Loss + Collision Cost
			loss = beta_d * KL + (0.7 * RCL + 0.3 * col_cost)
		
		return KL, RCL, loss
		
	# Forward Pass
	def forward(self, inp, traj_gt, init_state_ego, P_diag, Pdot, Pddot):

		# Lane Boundaries
		y_ub = inp[:, 0]
		y_lb = inp[:, 1]
		
		# Normalize input
		inp_norm = (inp - self.inp_mean) / self.inp_std
		
		# Mu & Variance
		mu, var = self.encode(inp_norm, traj_gt)
				
		# Sample from z -> Reparameterized 
		z = self.reparameterize(mu, var)
		
		# Decode y
		y_star = self.decode(z, inp_norm, init_state_ego)
		traj_sol = (P_diag @ y_star.T).T 

		# Collision cost
		col_cost = self.compute_col_cost(inp, y_star, traj_sol, Pdot, Pddot, y_ub, y_lb)
				
		return mu, var, traj_sol, col_cost
	
# Optimization Layer
class BatchOpt_DDN(LinEqConstDeclarativeNode):
	def __init__(self, P, Pdot, Pddot, num_batch):
		super().__init__(eps=1e-5, gamma=0.01)
 
		# P Matrices
		self.P = P.to(device)
		self.Pdot = Pdot.to(device)
		self.Pddot = Pddot.to(device)

		# Bernstein P
		self.P1 = self.P[0:25, :]
		self.P2 = self.P[25:50, :]
		self.P3 = self.P[50:75, :]
		self.P4 = self.P[75:100, :]

		self.Pdot1 = self.Pdot[0:25, :]
		self.Pdot2 = self.Pdot[25:50, :]
		self.Pdot3 = self.Pdot[50:75, :]
		self.Pdot4 = self.Pdot[75:100, :]
			
		self.Pddot1 = self.Pddot[0:25, :]
		self.Pddot2 = self.Pddot[25:50, :]
		self.Pddot3 = self.Pddot[50:75, :]
		self.Pddot4 = self.Pddot[75:100, :]
  
		# K constants
		self.k_p = torch.tensor(20.0, device=device)
		self.k_d = 2.0 * torch.sqrt(self.k_p)

		self.k_p_v = torch.tensor(20.0, device=device)
		self.k_d_v = 2.0 * torch.sqrt(self.k_p_v)  
  
		# Parameters
		self.v_max = 30.0 
		self.v_min = 0.1
		self.a_max = 8.0
		self.num_obs = 10
		self.num_batch = num_batch 
		self.steer_max = 0.6
		self.kappa_max = 0.230
		self.wheel_base = 2.5
		self.a_obs = 6.0
		self.b_obs = 3.2
		self.heading_max = torch.tensor(20 * np.pi/180, device=device)
		self.nvar = P.shape[1]
		self.num_partial = 25
		self.rho_v = 1.0
		self.rho_offset = 1.0
		self.num = 100
		t_fin = 15
		self.tot_time = torch.linspace(0, t_fin, self.num, device=device)
  
		# A & b Matrices
		self.A_eq_x = torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0]])
		self.A_eq_y = torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0]])

		# Smoothness
		self.weight_smoothness = 1.0 # 100.0
		self.cost_smoothness = self.weight_smoothness * torch.mm(self.Pddot.T, self.Pddot) # torch.eye(self.nvar, device=device)

	# Boundary Vectors
	def compute_boundary(self, initial_state_ego):
	 
		x_init_vec = torch.zeros([self.num_batch, 1], device=device) 
		y_init_vec = torch.zeros([self.num_batch, 1], device=device) 
  
		vx_init_vec = initial_state_ego[:, 2].reshape(self.num_batch, 1)
		vy_init_vec = initial_state_ego[:, 3].reshape(self.num_batch, 1)

		ax_init_vec = torch.zeros([self.num_batch, 1], device=device)
		ay_init_vec = torch.zeros([self.num_batch, 1], device=device)

		b_eq_x = torch.hstack([x_init_vec, vx_init_vec, ax_init_vec])
		b_eq_y = torch.hstack([y_init_vec, vy_init_vec, ay_init_vec])
	
		return b_eq_x, b_eq_y

	# Linear Constraint Parameters
	def linear_constraint_parameters(self, y):

		A = torch.block_diag(self.A_eq_x, self.A_eq_y)
		d = torch.hstack([self.b_eq_x, self.b_eq_y])
		
		return A, d
	
	# Objective Function
	def objective(self, *neural_output_batch, y):
	 
		cx = y[:, 0:self.nvar]
		cy = y[:, self.nvar:2*self.nvar]
		
		v_des_1 = neural_output_batch[1][:, 0]
		v_des_2 = neural_output_batch[1][:, 1]
		v_des_3 = neural_output_batch[1][:, 2]
		v_des_4 = neural_output_batch[1][:, 3]

		y_des_1 = neural_output_batch[1][:, 4]
		y_des_2 = neural_output_batch[1][:, 5]
		y_des_3 = neural_output_batch[1][:, 6]
		y_des_4 = neural_output_batch[1][:, 7]
  
		# A & b Matrices
		A_pd_1 = self.Pddot1 - self.k_p * self.P1 - self.k_d * self.Pdot1
		b_pd_1 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_1)[:, None]

		A_pd_2 = self.Pddot2 - self.k_p * self.P2 - self.k_d * self.Pdot2
		b_pd_2 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_2)[:, None]
			
		A_pd_3 = self.Pddot3 - self.k_p * self.P3 - self.k_d * self.Pdot3
		b_pd_3 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_3)[:, None]

		A_pd_4 = self.Pddot4 - self.k_p * self.P4 - self.k_d * self.Pdot4
		b_pd_4 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_4)[:, None]

		A_vd_1 = self.Pddot1 - self.k_p_v * self.Pdot1 - self.k_d_v * self.Pddot1
		b_vd_1 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_1)[:, None]

		A_vd_2 = self.Pddot2 - self.k_p_v * self.Pdot2 - self.k_d_v * self.Pddot2
		b_vd_2 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_2)[:, None]

		A_vd_3 = self.Pddot3 - self.k_p_v * self.Pdot3 - self.k_d_v * self.Pddot3
		b_vd_3 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_3)[:, None]

		A_vd_4 = self.Pddot4 - self.k_p_v * self.Pdot4 - self.k_d_v * self.Pddot4
		b_vd_4 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_4)[:, None]
  
		cost_pd_1 = torch.linalg.norm(torch.mm(A_pd_1, cy.T).T - b_pd_1, dim = 1)
		cost_pd_2 = torch.linalg.norm(torch.mm(A_pd_2, cy.T).T - b_pd_2, dim = 1)
		cost_pd_3 = torch.linalg.norm(torch.mm(A_pd_3, cy.T).T - b_pd_3, dim = 1)
		cost_pd_4 = torch.linalg.norm(torch.mm(A_pd_4, cy.T).T - b_pd_4, dim = 1)
  
		cost_vd_1 = torch.linalg.norm(torch.mm(A_vd_1, cx.T).T - b_vd_1, dim = 1)
		cost_vd_2 = torch.linalg.norm(torch.mm(A_vd_2, cx.T).T - b_vd_2, dim = 1)
		cost_vd_3 = torch.linalg.norm(torch.mm(A_vd_3, cx.T).T - b_vd_3, dim = 1)
		cost_vd_4 = torch.linalg.norm(torch.mm(A_vd_4, cx.T).T - b_vd_4, dim = 1)
  
		cost_smoothness_x = torch.linalg.norm(torch.mm(self.Pddot, cx.T).T, dim = 1)
		cost_smoothness_y = torch.linalg.norm(torch.mm(self.Pddot, cy.T).T, dim = 1)
	
		# Inner Cost
		inner_cost_x = self.weight_smoothness * cost_smoothness_x**2 + self.rho_v * (cost_vd_1**2 + cost_vd_2**2 + cost_vd_3**2 + cost_vd_4**2)
		inner_cost_y = self.weight_smoothness * cost_smoothness_y**2 + self.rho_offset * (cost_pd_1**2 + cost_pd_2**2 + cost_pd_3**2 + cost_pd_4**2)
		inner_cost = 0.5 * inner_cost_x + 0.5 * inner_cost_y
  
		return inner_cost

	# Solve Function
	def solve(self, initial_state_ego, neural_output_batch):
	 
		self.b_eq_x, self.b_eq_y = self.compute_boundary(initial_state_ego) 

		v_des_1 = neural_output_batch[:, 0]
		v_des_2 = neural_output_batch[:, 1]
		v_des_3 = neural_output_batch[:, 2]
		v_des_4 = neural_output_batch[:, 3]

		y_des_1 = neural_output_batch[:, 4]
		y_des_2 = neural_output_batch[:, 5]
		y_des_3 = neural_output_batch[:, 6]
		y_des_4 = neural_output_batch[:, 7]

		# A & b Matrices
		A_pd_1 = self.Pddot1 - self.k_p * self.P1 - self.k_d * self.Pdot1
		b_pd_1 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_1)[:, None]

		A_pd_2 = self.Pddot2 - self.k_p * self.P2 - self.k_d * self.Pdot2
		b_pd_2 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_2)[:, None]
			
		A_pd_3 = self.Pddot3 - self.k_p * self.P3 - self.k_d * self.Pdot3
		b_pd_3 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_3)[:, None]

		A_pd_4 = self.Pddot4 - self.k_p * self.P4 - self.k_d * self.Pdot4
		b_pd_4 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_4)[:, None]

		A_vd_1 = self.Pddot1 - self.k_p_v * self.Pdot1 - self.k_d_v * self.Pddot1
		b_vd_1 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_1)[:, None]

		A_vd_2 = self.Pddot2 - self.k_p_v * self.Pdot2 - self.k_d_v * self.Pddot2
		b_vd_2 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_2)[:, None]

		A_vd_3 = self.Pddot3 - self.k_p_v * self.Pdot3 - self.k_d_v * self.Pddot3
		b_vd_3 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_3)[:, None]

		A_vd_4 = self.Pddot4 - self.k_p_v * self.Pdot4 - self.k_d_v * self.Pddot4
		b_vd_4 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_4)[:, None]
  
		# Cost
		cost_x = self.cost_smoothness + self.rho_v * torch.mm(A_vd_1.T, A_vd_1) + self.rho_v * torch.mm(A_vd_2.T, A_vd_2) + self.rho_v * torch.mm(A_vd_3.T, A_vd_3) + self.rho_v * torch.mm(A_vd_4.T, A_vd_4)
		cost_y = self.cost_smoothness + self.rho_offset * torch.mm(A_pd_1.T, A_pd_1) + self.rho_offset * torch.mm(A_pd_2.T, A_pd_2) + self.rho_offset * torch.mm(A_pd_3.T, A_pd_3) + self.rho_offset * torch.mm(A_pd_4.T, A_pd_4)
  
		cost_mat_x = torch.vstack([torch.hstack([cost_x, self.A_eq_x.T]), torch.hstack([self.A_eq_x, torch.zeros([self.A_eq_x.shape[0], self.A_eq_x.shape[0]], device=device)])])
		cost_mat_y = torch.vstack([torch.hstack([cost_y, self.A_eq_y.T]), torch.hstack([self.A_eq_y, torch.zeros([self.A_eq_y.shape[0], self.A_eq_y.shape[0]], device=device)])])

		lincost_x = -self.rho_v * torch.mm(A_vd_1.T, b_vd_1.T).T - self.rho_v * torch.mm(A_vd_2.T, b_vd_2.T).T - self.rho_v * torch.mm(A_vd_3.T, b_vd_3.T).T - self.rho_v * torch.mm(A_vd_4.T, b_vd_4.T).T
		lincost_y = -self.rho_offset * torch.mm(A_pd_1.T, b_pd_1.T).T - self.rho_offset * torch.mm(A_pd_2.T, b_pd_2.T).T - self.rho_offset * torch.mm(A_pd_3.T, b_pd_3.T).T - self.rho_offset * torch.mm(A_pd_4.T, b_pd_4.T).T

		# Solution
		sol_x = torch.linalg.solve(cost_mat_x, torch.hstack([-lincost_x, self.b_eq_x]).T).T
		sol_y = torch.linalg.solve(cost_mat_y, torch.hstack([-lincost_y, self.b_eq_y]).T).T
		
		c_x = sol_x[:, 0:self.nvar]
		c_y = sol_y[:, 0:self.nvar]

		dual_x = sol_x[:, self.nvar:]
		dual_y = sol_y[:, self.nvar:]

		# Solution & Context
		y = torch.hstack([c_x, c_y])
		nu = torch.hstack([dual_x, dual_y])
  
		ctx = {"nu" : nu}

		return y, ctx    

class DeclarativeFunction(torch.autograd.Function):
	"""Generic declarative autograd function.
	Defines the forward and backward functions. Saves all inputs and outputs,
	which may be memory-inefficient for the specific problem.
	
	Assumptions:
	* All inputs are PyTorch tensors
	* All inputs have a single batch dimension (b, ...)
	"""
	@staticmethod
	def forward(ctx, problem, *inputs):
		output, solve_ctx = torch.no_grad()(problem.solve)(*inputs)
		ctx.save_for_backward(output, *inputs)
		ctx.problem = problem
		ctx.solve_ctx = solve_ctx
		return output.clone()

	@staticmethod
	def backward(ctx, grad_output):
		output, *inputs = ctx.saved_tensors
		problem = ctx.problem
		solve_ctx = ctx.solve_ctx
		output.requires_grad = True
		inputs = tuple(inputs)
		grad_inputs = problem.gradient(*inputs, y=output, v=grad_output,
			ctx=solve_ctx)
		return (None, *grad_inputs)

class DeclarativeLayer(torch.nn.Module):
	"""Generic declarative layer.
	
	Assumptions:
	* All inputs are PyTorch tensors
	* All inputs have a single batch dimension (b, ...)
	Usage:
		problem = <derived class of *DeclarativeNode>
		declarative_layer = DeclarativeLayer(problem)
		y = declarative_layer(x1, x2, ...)
	"""
	def __init__(self, problem):
		super(DeclarativeLayer, self).__init__()
		self.problem = problem
		
	def forward(self, *inputs):
		return DeclarativeFunction.apply(self.problem, *inputs)