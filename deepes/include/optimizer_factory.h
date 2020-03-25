#ifndef OPTIMIZER_FACTORY_H
#define OPTIMIZER_FACTORY_H

#include <algorithm>
#include <memory>
#include "optimizer.h"
#include "sgd_optimizer.h"
#include "adam_optimizer.h"
#include "deepes.pb.h"

namespace DeepES{
/* @brief: create an optimizer according to the configuration"
 * @args: 
 *    config: configuration for the optimizer
 * 
 */
std::shared_ptr<Optimizer> create_optimizer(const OptimizerConfig& optimizer_config);

}//namespace

#endif
