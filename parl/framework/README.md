## Parameters management in PARL

In RL, we usually need to reuse parameters of layers and (periodically) synchronize parameters betweent models.

PaddlePaddle allows users to reuse parameters by specifying the same custom name to parameter attributes. This way
is flexible, but it will make the code complicated when needed to reuse the parameter in multiple places or reuse 
all parameters in a module.

And when synchronizing parameters between two models in PaddlePaddle, we have to name the parameters for every layer 
in the models, and establish the parameters synchronization mapping according to the parameters names, 
which also makes the code complicated.

To make PaddlePaddle more convenient to implement the reinforcement learning algorithms, PARL framework provides an easier
mechanism to resue parameters of layers and an API to synchronize parameters between models directly.

### Reuse parameters of layers in PARL

PARL use a `LayerFunc` object to wrap every layer with parameters in PaddlePaddle，which will manage parameters 
of layers automatically. 

When you declare a layer with `parl.layers`, framework will help manage parameters and return a callable `LayerFunc` 
object to you. And every time you want to reuse the parameters, you only need to call the corresponding `LayerFunc` object 
but without specifying the parameters names. 

Here is an example:

```python
import parl.layers as layers

class MLPModel(Model):
    def __init__(self):
        self.fc = layers.fc(size=64) # automatically create parameters names "fc_0.w" and "fc_0.b"

    def policy1(self, obs):
        out = self.fc(obs) # Really create parameters with parameters names "fc_0.w" and "fc_0.b"
        ...
    
    def policy2(self, obs):
        out = self.fc(obs) # Reusing parameters
        ...
```

### Synchronize parameters between models in PARL

PARL provides an API `sync_params_to` in `parl.Model` to synchronize its all parameters to another model. 
When you need synchronize parameters bewteen two models, only following steps needed:

1. Construct a model inheriting `parl.Model`, and declare layers needing synchronization with `parl.layers` in the model.
2. Deepcopy a target model. 
3. After parameters initialized, Call `sync_params_to` API to synchronize all parameters in the model to target model.

Here is an example:
```python
import parl.layers as layers
import parl.Model as Model

class MLPModel(Model):
    def __init__(self):
        self.fc = layers.fc(size=64)

    def policy(self, obs):
        out = self.fc(obs)
        return out
        
model = MLPModel() 
target_model = deepcopy(model) # automatically create new unique parameters names for target_model.fc

# build program
x = layers.data(name='x', shape=[100], dtype="float32")
y1 = model.policy(x) 
y2 = target_model.policy(x)  

...
# Need initialize parameters before calling sync_params_to
fluid_executor.run(fluid.default_startup_program()) 
...

# synchronize parameters
model.sync_params_to(target_model, gpu_id=gpu_id)

```

By the way, `parl.Model` also provides a helpful property `parameter_names`, which can return all parameters names 
declared with `parl.layers` in the model.
