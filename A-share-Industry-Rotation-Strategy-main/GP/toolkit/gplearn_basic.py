"""gplearn基础用法示例"""
import numpy as np
import pandas as pd
from gplearnTiming import fitness
from gplearnTiming.genetic import SymbolicRegressor
from datetime import datetime


def score_func_basic(y, y_pred, sample_weight, **args):
    return sum((pd.Series(y_pred) - y) ** 2)


m = fitness.make_fitness(function=score_func_basic, greater_is_better=False, wrap=False)

cmodel_gp = SymbolicRegressor(
    population_size=500, generations=10, metric=m, tournament_size=50,
    function_set=('add', 'sub', 'mul', 'abs', 'neg', 'sin', 'cos', 'tan'),
    const_range=(-1.0, 1.0), parsimony_coefficient='auto',
    init_depth=(2, 4), init_method='half and half',
    p_crossover=0.2, p_subtree_mutation=0.2,
    p_hoist_mutation=0.2, p_point_mutation=0.2, p_point_replace=0.2,
    max_samples=1.0, feature_names=None, warm_start=False,
    low_memory=False, n_jobs=1, verbose=1, random_state=0
)


if __name__ == '__main__':
    start = datetime.now()

    X1 = pd.DataFrame({'a': range(1000), 'b': np.random.randint(-10, 10, 1000)})
    Y1 = X1.sum(axis=1)
    print('Target: Y1 = X1.sum(axis=1)')
    cmodel_gp.fit(X1, Y1)
    print(cmodel_gp)

    X2 = pd.DataFrame({'a': range(1000), 'b': np.random.randint(0, 10, 1000)})
    Y2 = np.cos(X2['a']) - np.sin(X2['b'])
    cmodel_gp.fit(X2, Y2)
    print(cmodel_gp)

    print(f'Time: {datetime.now() - start}')
