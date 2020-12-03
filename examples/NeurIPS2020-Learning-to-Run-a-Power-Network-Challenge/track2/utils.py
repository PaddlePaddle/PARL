from __future__ import print_function
import sys
import numpy as np


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


selected_feature = [
    'loads_q', 'loads_v', 'prods_q', 'prods_v', 'rho', 'line_status',
    'hour_of_day', 'month'
]
inference_info = np.load('./saved_files/inference_info.npz')
col = inference_info['col']
mean = inference_info['mean']
std = inference_info['std']

def process(raw_obs):
    obs = raw_obs.to_dict()
    x = dict()
    x['loads_p'] = obs['loads']['p']
    x['loads_q'] = obs['loads']['q']
    x['loads_v'] = obs['loads']['v']
    x['prods_p'] = obs['prods']['p']
    x['prods_q'] = obs['prods']['q']
    x['prods_v'] = obs['prods']['v']
    x['lines_or_p'] = obs['lines_or']['p']
    x['lines_or_q'] = obs['lines_or']['q']
    x['lines_or_v'] = obs['lines_or']['v']
    x['lines_or_a'] = obs['lines_or']['a']
    x['lines_ex_p'] = obs['lines_ex']['p']
    x['lines_ex_q'] = obs['lines_ex']['q']
    x['lines_ex_v'] = obs['lines_ex']['v']
    x['lines_ex_a'] = obs['lines_ex']['a']
    x['day_of_week'] = raw_obs.day_of_week
    x['month'] = raw_obs.month
    x['hour_of_day'] = raw_obs.hour_of_day
    to_maintain_lines = np.where((raw_obs.time_next_maintenance>0) \
                          & (raw_obs.time_next_maintenance<2))[0]
    x['rho'] = np.copy(obs['rho'])
    x['line_status'] = np.copy(obs['line_status'].astype(float))
    line_num = x['line_status'].shape[0]
    if len(to_maintain_lines):
        x['rho'][to_maintain_lines] = 0.0
        x['line_status'][to_maintain_lines] = 0.0
    x['line_status'] += np.array([x * 2 for x in range(line_num)])
    x['rho'] = x['rho'] - 1.0
    data = []
    for feature in selected_feature:
        col_data = x[feature]
        if isinstance(col_data, np.int32):
            col_data = np.array([col_data])
        data.append(col_data)
    data = np.concatenate(data)
    data = data[col]
    assert data.shape[0] == mean.shape[0]
    assert data.shape[0] == std.shape[0]
    data = (data - mean) / std
    return data
