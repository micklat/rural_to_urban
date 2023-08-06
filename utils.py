import numpy as np
import pandas as pd

def print_quantiles(samples, quantiles=np.arange(0.1,0.9,0.1)):
    samples = samples.ravel()
    print(dict(zip(quantiles, np.quantile(samples, quantiles)))) 


pasted_squiggle_results = """
Name	Mean	Stdev	5%	10%	25%	50%	75%	90%	95%
charity$/doubling	54	46	14	18	27	42	65	100	130
charity$/DALY	120	110	33	42	61	96	150	230	310
total$/doubling	54	46	14	18	27	42	65	100	130
total$/DALY	120	110	33	42	61	96	150	230	310
"""

def parse_squiggle():
    rows = pasted_squiggle_results.split('\n')
    def parse_row(row):
        return [s.lower() for s in row.split("\t") if s]
    columns = []
    values = {}
    for row in rows:
        parts = parse_row(row)
        if not parts:
            continue
        if not columns:
            columns = parts[1:]
        else:
            name = parts[0]
            values[name] = dict(zip(columns, map(float, parts[1:])))
    return pd.DataFrame(values)

squiggle_results = parse_squiggle()

def calc_cost_effectiveness(samples):
    results_arg_names = \
            "charity_cost, total_cost, effect_in_doubling_consumption, effect_in_DALYs"
    results_arg_names = results_arg_names.split(", ")

    charity_cost = samples.total_charity_cost
    total_cost = samples.total_cost
    effect_in_doubling_consumption = samples.total_consumption_doublings
    effect_in_DALYs = samples.total_DALYs

    def cost_effectiveness(cost, effect): return cost.values.ravel()/effect.values.ravel()

    dists = {
            "charity$/doubling": cost_effectiveness(charity_cost, effect_in_doubling_consumption),
            "charity$/DALY": cost_effectiveness(charity_cost, effect_in_DALYs),
            "total$/doubling": cost_effectiveness(total_cost, effect_in_doubling_consumption),
            "total$/DALY": cost_effectiveness(total_cost, effect_in_DALYs),
    }
    for ratio_name, values in dists.items():
        squiggle_vals = squiggle_results[ratio_name.lower()]
        for statistic in squiggle_vals.index.values:
            if statistic.endswith('%'):
                f_statistic = lambda arr: np.quantile(arr, float(statistic[:-1])/100)
            else:
                f_statistic = {"mean": np.mean, "stdev": np.std}[statistic]
            print(ratio_name, statistic, squiggle_vals[statistic], f_statistic(values))


    

