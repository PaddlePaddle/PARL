#include <torch/torch.h>

const double kPi = 3.1415926535898;

class CartPole {
	// Translated from openai/gym's cartpole.py
public:
	double gravity = 9.8;
	double masscart = 1.0;
	double masspole = 0.1;
	double total_mass = (masspole + masscart);
	double length = 0.5; // actually half the pole's length;
	double polemass_length = (masspole * length);
	double force_mag = 10.0;
	double tau = 0.02; // seconds between state updates;

	// Angle at which to fail the episode
	double theta_threshold_radians = 12 * 2 * kPi / 360;
	double x_threshold = 2.4;
	int steps_beyond_done = -1;

	torch::Tensor state;
	double reward;
	bool done;
	int step_ = 0;

	torch::Tensor getState() {
		return state;
	}

	double getReward() {
		return reward;
	}

	double isDone() {
		return done;
	}

	void reset() {
		state = torch::empty({ 4 }).uniform_(-0.05, 0.05);
		steps_beyond_done = -1;
		step_ = 0;
	}

	CartPole() {
		reset();
	}

	void step(int action) {
		auto x = state[0].item<float>();
		auto x_dot = state[1].item<float>();
		auto theta = state[2].item<float>();
		auto theta_dot = state[3].item<float>();

		auto force = (action == 1) ? force_mag : -force_mag;
		auto costheta = std::cos(theta);
		auto sintheta = std::sin(theta);
		auto temp = (force + polemass_length * theta_dot * theta_dot * sintheta) /
			total_mass;
		auto thetaacc = (gravity * sintheta - costheta * temp) /
			(length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
		auto xacc = temp - polemass_length * thetaacc * costheta / total_mass;

		x = x + tau * x_dot;
		x_dot = x_dot + tau * xacc;
		theta = theta + tau * theta_dot;
		theta_dot = theta_dot + tau * thetaacc;
		state = torch::tensor({ x, x_dot, theta, theta_dot });

		done = x < -x_threshold || x > x_threshold ||
			theta < -theta_threshold_radians || theta > theta_threshold_radians ||
			step_ > 200;

		if (!done) {
			reward = 1.0;
		}
		else if (steps_beyond_done == -1) {
			// Pole just fell!
			steps_beyond_done = 0;
			reward = 0;
		}
		else {
			if (steps_beyond_done == 0) {
				AT_ASSERT(false); // Can't do this
			}
		}
		step_++;
	}
};
