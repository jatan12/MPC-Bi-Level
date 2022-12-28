import numpy as np
import jax.numpy as jnp
import jax
from jax import random
from expert_mpc.expert_bilevel import batch_opt_nonhol

class ExpertPolicy(batch_opt_nonhol):
    """
    Expert deterministic policy running the MPC
    """

    def __init__(self, Wandb, BN, inp_mean, inp_std, use_nn=True):
        super().__init__(Wandb, BN, inp_mean, inp_std)
        num = 100
        v_max = 30.0
        a_max = 8.0
        num_batch = 1000 # 250, 500, 750, 1000
        self.lamda_x = jnp.zeros((num_batch,  self.nvar))
        self.lamda_y = jnp.zeros((num_batch,  self.nvar))
        self.d_a = a_max * jnp.ones((num_batch, num))
        self.alpha_a = jnp.zeros((num_batch, num))
        self.alpha_v = jnp.zeros((num_batch, num))
        self.d_v = v_max * jnp.ones((num_batch, num))
        self.s_lane = jnp.zeros((num_batch, 2*num))

        self.use_nn = use_nn
        key = random.PRNGKey(0)
        self.key = key
        
        mean_vx_1 = 5
        mean_vx_2 = 5
        mean_vx_3 = 5
        mean_vx_4 = 5

        mean_y_des_1 = 0
        mean_y_des_2 = 0
        mean_y_des_3 = 0
        mean_y_des_4 = 0
        
        self.mean_param = jnp.hstack(( mean_vx_1, mean_vx_2, mean_vx_3, mean_vx_4, mean_y_des_1, mean_y_des_2, mean_y_des_3, mean_y_des_4   ))
        self.diag_param = np.hstack(( v_max, v_max, v_max, v_max, v_max, 46.0, 46.0, 46.0  ))
        self.cov_param = jnp.asarray(np.diag(self.diag_param)) 
    
    def predict(self, obs, ax, ay, v_des):
        x = 0
        y = 0
        ub = obs[0]
        lb = obs[1]
        vx = obs[2]
        vy = obs[3]

        initial_state = jnp.hstack((x, y, vx, vy, ax, ay))
        x_obs_temp = obs[5::5]
        y_obs_temp = obs[6::5]
        vx_obs = obs[7::5]
        vy_obs = obs[8::5]

        # Observation of shape (1, 55)
        inp = obs.reshape(1, -1)

        if self.use_nn:

            key, subkey = random.split(self.key)

            # Normalized Input to the network
            inp_norm = (inp - self.inp_mean) / self.inp_std
            batch_inp_norm = jnp.vstack([inp_norm] * (self.decode_batch_size - self.ellite_num))

            # z ~ N(0, 1)
            z = random.normal(key, ((self.decode_batch_size - self.ellite_num), self.z_dim)) 
            batch_inputs = jnp.concatenate([z, batch_inp_norm], axis = 1)

            # Generate y's from the decoder
            neural_output_warmstart = self.jax_decoder(batch_inputs)
            neural_output_warmstart = neural_output_warmstart.at[:, 0:4].set(jax.nn.sigmoid(neural_output_warmstart[:, 0:4]) * 27. + 3.)
            neural_output_warmstart = neural_output_warmstart[0:self.ellite_num]

        else:
            neural_output_warmstart = self.sampling_param(lb, ub, self.mean_param, self.cov_param)
            neural_output_warmstart = neural_output_warmstart[0:self.ellite_num]
      
        x_obs, y_obs = self.compute_obs_trajectories(x_obs_temp, y_obs_temp, vx_obs, vy_obs, x, y)
        
        if self.use_nn:
            c_x_best, c_y_best, self.mean_param, neural_output_warmstart  = self.compute_cem_nn(inp, initial_state, self.lamda_x, self.lamda_y, x_obs, y_obs, lb, ub,  
                                                                                            self.alpha_a, self.d_a, self.alpha_v, self.d_v, v_des, self.s_lane, 
                                                                                            self.mean_param, self.cov_param, neural_output_warmstart)
        else:    
            c_x_best, c_y_best, self.mean_param, neural_output_warmstart  = self.compute_cem(initial_state, self.lamda_x, self.lamda_y, x_obs, y_obs, lb, ub,  
                                                                                            self.alpha_a, self.d_a, self.alpha_v, self.d_v, v_des, self.s_lane, 
                                                                                            self.mean_param, self.cov_param, neural_output_warmstart)
        
        return np.hstack((np.array(c_x_best), np.array(c_y_best)))