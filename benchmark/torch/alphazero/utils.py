class dotdict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

def win_loss_draw(score):
    if score>0: 
        return 'win'
    if score<0: 
        return 'loss'
    return 'draw'

"""
split one list to multiple lists
"""
split_group = lambda the_list, group_size: zip(*(iter(the_list),) * group_size)
