#ifndef OPTIMIZER_H
#define OPTIMIZER_H
namespace DeepES{

class Optimizer {
public:
  Optimizer() : _base_lr(1e-3), _update_times(0) {}
  Optimizer(float base_lr) : _base_lr(base_lr), _update_times(0) {}
  template<typename T>
  bool update(T weights, float* gradient, int size, std::string param_name="") {
    bool success = true;
    ++_update_times;
    compute_step(gradient, size, param_name);
    for (int i = 0; i < size; ++i) {
      weights[i] -= _base_lr * gradient[i];
    }
    return success;
  } // template function

protected:
  virtual void compute_step(float* graident, int size, std::string param_name="") = 0;
  float _base_lr;
  float _update_times;
};

class SGDOptimizer: public Optimizer {
public:
  SGDOptimizer(float base_lr, float momentum=0.0):Optimizer(base_lr), _momentum(momentum) {}

protected:
  void compute_step(float* gradient, int size, std::string param_name="") {
  }

private:
  float _momentum;

}; //namespace

//class AdamOptimizer: public Optimizer {
//public:
//  AdamOptimizer(float base)
//}

};
#endif
