#include "optimizer_factory.h"

namespace DeepES{

std::shared_ptr<Optimizer> create_optimizer(const OptimizerConfig& optimizer_config) {
  std::shared_ptr<Optimizer> optimizer;
  std::string opt_type = optimizer_config.type();
  std::transform(opt_type.begin(), opt_type.end(), opt_type.begin(), ::tolower);
  if (opt_type == "sgd") {
    optimizer = std::make_shared<SGDOptimizer>(optimizer_config.base_lr(), \
                                                optimizer_config.momentum());
  }else if (opt_type == "adam") {
    optimizer = std::make_shared<AdamOptimizer>(optimizer_config.base_lr(), \
                                                  optimizer_config.beta1(), \
                                                  optimizer_config.beta2(), \
                                                  optimizer_config.epsilon());
  }else {
    // TODO: NotImplementedError
  }
  return optimizer;
}

}//namespace
