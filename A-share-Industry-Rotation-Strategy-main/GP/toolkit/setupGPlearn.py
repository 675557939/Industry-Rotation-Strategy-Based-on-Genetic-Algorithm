from gplearn.genetic import SymbolicRegressor
from gplearn import fitness
import dill


def gp_save_factor(model, factor_num=''):
    try:
        with open(f'./result/factor/factor{factor_num}.pickle', 'wb') as f:
            dill.dump(model, f)
    except Exception as e:
        print(f"保存失败: {e}")


def my_gplearn(function_set, score_func, pop_num=1000, gen_num=10, tour_num=20,
               random_state=42, feature_names=None, verbose=1, n_jobs=1, **kwargs):
    metric = fitness.make_fitness(function=score_func, greater_is_better=True, wrap=False)

    return SymbolicRegressor(
        population_size=pop_num,
        generations=gen_num,
        metric=metric,
        tournament_size=tour_num,
        function_set=function_set,
        const_range=(-1.0, 1.0),
        parsimony_coefficient=kwargs.get('parsimony_coefficient', 'auto'),
        max_samples=kwargs.get('max_samples', 1.0),
        stopping_criteria=100.0,
        init_depth=kwargs.get('init_depth', (2, 4)),
        init_method='half and half',
        p_crossover=0.8,
        p_subtree_mutation=0.05,
        p_hoist_mutation=0.02,
        p_point_mutation=0.05,
        p_point_replace=0.03,
        feature_names=feature_names,
        warm_start=False,
        low_memory=False,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state
    )
