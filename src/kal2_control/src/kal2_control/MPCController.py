import casadi as ca
import numpy as np

class MPCController:
    def __init__(self,):
        self.delta_t = 0.1
        self.prediction_horizon = 30

        self.v_max = 1.0
        self.v_min = 0.0
        self.v_ref = 0.5
        self.acc_max = 0.1
        self.acc_min = -0.1
        self.delta_ref = 0.0
        self.delta_max = np.pi / 4
        self.delta_min = -self.delta_max
        self.acc_ref = 0.0

        self.L = 0.29 # Wheelbase
        self.Q = np.diag([2.0, 2.0, 5.0, 2.5])
        self.R = np.diag([2.0, 2.5])
        self.S = np.diag([5.0, 5.0])


        self.next_states = np.zeros((self.prediction_horizon + 1, 4))
        self.u0 = np.zeros((self.prediction_horizon, 2))
        self.setup_controller()
    
    def setup_controller(self):
        self.opti = ca.Opti()

        # State variables matrix
        self.opt_states = self.opti.variable(self.prediction_horizon + 1, 4)
        x = self.opt_states[:, 0]
        y = self.opt_states[:, 1]
        theta = self.opt_states[:, 2]
        v = self.opt_states[:, 3]
        

        # Control variables matrix
        self.opt_controls = self.opti.variable(self.prediction_horizon, 2)
        delta = self.opt_controls[:, 0]
        acc = self.opt_controls[:, 1]

        # System dynamics
        f = lambda x, u: ca.vertcat(x[3] * ca.cos(x[2]), x[3] * ca.sin(x[2]), x[3] * ca.tan(u[0]) / self.L, u[1])

        # Define the parameters for the initial state
        self.v_ref = self.opti.parameter(self.prediction_horizon, 1)
        self.opt_controls_ref = self.opti.parameter(self.prediction_horizon, 2)
        self.opt_states_ref = self.opti.parameter(self.prediction_horizon, 4)

        # Initial state constraints
        self.opti.subject_to(self.opt_states[0, :] == self.opt_states_ref[0, :])
        for i in range(self.prediction_horizon):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T * self.delta_t
            self.opti.subject_to(self.opt_states[i + 1, :] == x_next)

        # Define the cost function
        J = 0
        steering_change = self.opt_controls[1:, 0] - self.opt_controls[:-1, 0]
        # velocity_change = self.opt_controls[1:, 1] - self.opt_controls[:-1, 1]
        yaw_change = self.opt_states[1:, 2] - self.opt_states[:-1, 2]

        for i in range(self.prediction_horizon):
            state_error = self.opt_states[i, :] - self.opt_states_ref[i, :]
            control_error = self.opt_controls[i, :] - self.opt_controls_ref[i, :]

            if i < self.prediction_horizon - 1:
                control_change = self.opt_controls[i + 1, :] - self.opt_controls[i, :]
                J = J + ca.mtimes([control_change, self.S, control_change.T])

            J = J + ca.mtimes([state_error, self.Q, state_error.T]) + ca.mtimes([control_error, self.R, control_error.T]) + 5.0 * ca.sumsqr(yaw_change) + 2.0 * ca.sumsqr(steering_change)


        terminal_state_error = self.opt_states[-1, :] - self.opt_states_ref[-1, :]
        J = J + ca.mtimes([terminal_state_error, self.Q, terminal_state_error.T])
            
        self.opti.minimize(J)
        # Define the input constraints
        self.opti.subject_to(self.opti.bounded(self.v_min, v, self.v_max))
        self.opti.subject_to(self.opti.bounded(-0.1, steering_change, 0.1))
        # Define the output constraints
        self.opti.subject_to(self.opti.bounded(self.acc_min, acc, self.acc_max))
        self.opti.subject_to(self.opti.bounded(self.delta_min, delta, self.delta_max))
        
        # Define the solver
        opts = {'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-6, 'ipopt.acceptable_obj_change_tol': 1e-6}
        self.opti.solver('ipopt', opts)

    def calculate_trajectory(self, opt_states_now):
        
        # Set the initial state and reference trajectory
        self.opti.set_value(self.opt_states_ref, opt_states_now)
        self.opti.set_value(self.opt_controls_ref, np.zeros((self.prediction_horizon, 2)))
        
        self.opti.set_initial(self.opt_controls, self.u0)
        self.opti.set_initial(self.opt_states, self.next_states)

        # Solve the optimization problem
        solution = self.opti.solve()

        # Set the initial guess for the next iteration
        self.opti.set_initial(solution.value_variables())

        # obtain the control inputs and the next states
        self.u0 = solution.value(self.opt_controls)
        self.next_states = solution.value(self.opt_states)
        
        print(f"control output: {self.u0}")
        print(f"next states shape: {self.next_states.shape}")

        return self.u0[0, :]
        