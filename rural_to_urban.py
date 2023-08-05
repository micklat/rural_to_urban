#!/usr/bin/env python

# A python port of Edo's squiggle model from:
# https://squigglehub.org/users/edoarad/models/ce_migration_estimate

from importlib import reload
import sys
sys.path.extend([
    "/home/michael/Documents/programming/packages/nbag", 
    "/home/michael/Documents/programming/packages/nba_pymc",
    "/home/michael/Documents/programming/packages/credints"])

import nba_pymc 
from nba_pymc import beta, Model
from nba_pymc.math import log, minimum
from credints import EqualTailIntervals
import numpy as np
from typing import Mapping


ci = EqualTailIntervals(nba_pymc)

m = Model()
m.__enter__()

K=1000
M=10**6

def log2(x):
    return log(x) / log(2)

# -------------------------
# Key parameters and assumptions [Rwanda]
# -------------------------

# Intervention timing
num_years = 50 #for calculations involving npv 
preperation_time = 1 #years - time needed before starting the intervention 
cover_time = 4 #years - We assume that the charity would run the campaign in all target regions over the scope of 4 years, i.e. 25% of the regions in each year
run_time_unsuccessful = 3 #years - How long charity runs if unsuccessful
run_time_successful = 5 #years - How long charity program runs if successful

# Timing of behavior and effect
avg_delay_of_behavior = 1 #years - Average delay in behavior change
migrate_return_ratios = {
  "return without finding job": 0.3, # Based on Baseler (2022)'s figure for 70% of migrants having a job in the past week
  "return after 1 year": 0.3, # A guess
  "return after 2 years": 0.1, # Baseler (2022) observe that 40% of migrants stay in Nairobi for at least two years; I'm splitting it over this row and the next
  "stay indefinitely": 0.3,
}

rise_in_effect_each_year = beta(8 * 1, 8 * 19) # This is a beta distribution with odds ratio of 1:19, and the multiplier of 8 is to get to the estimated [3%,8%] CI
years_before_plateau = ci.log_normal(0.9, (4, 12)) # number of years before the effect stops increasing.

# interesting to note that the way the above couple of lines will be used is in modeling epistemic uncertainty, 
# not aleatoric (statistical) uncertainty.
# That is, the model assumes all people in the population are affected by the intervention in exactly the same 
# way, but we don't know what that way is. We can model this differently if we want.
# Key data
avg_consumption = 1004 #USD per year - Estimate, based on the fact that 42% of rural dwellers live in extreme poverty (<$2.15). The median earner must be bit above this figure, and the average earner will be a bit higher still (due to the skewness of the income distribution). My guess is $2.75 per day. I assume that virtually all of this is consumed.
pop_growth_rate = 2.58 / 100 #per year - This has been quite stable for decades
population = 13.94*M #Population
percent_population_rural = 14 / 100 #Percentage of population in rural (and non-periurban) areas
rural_population = population * percent_population_rural #Rural population
avg_household_size = 4.3 #Average household size
# Key assumptions
discount_rate = beta(10 * 4, 10 * 96) #  epistemic uncertainty
chance_change_happen_anyway = beta(12 * 1.5, 12 * 98.5)
prob_of_success = beta(5 * 1, 5 * 1)
consumption_doubling_to_DALY_ratio = 1/2.3 # DALYs per doubling of consumption
num_years = 50 #for calculations involving npv 
preperation_time = 1 #years - time needed before starting the intervention 
cover_time = 4 #years - We assume that the charity would run the campaign in all target regions over the scope of 4 years, i.e. 25% of the regions in each year
run_time_unsuccessful = 3 #years - How long charity runs if unsuccessful
run_time_successful = 5 #years - How long charity program runs if successful

years = np.arange(1, num_years+1)

# ## FIXED CHARITY COSTS
overhead_year_1 = 125*K # USD - Overhead in year 1
overhead_scale = 225*K #USD - Overhead at scale

# ## VARIABLE CHARITY COSTS
# Basic data for cost calcualtion
num_radio_stations = 4 #How many different radio stations will this run on each year at scale
num_ads_per_day_per_station = 5 #Number of ads per day per station
# Radio advertising – airtime costs
# BTW this looks like a bug
cost_per_60_ads = ci.log_normal(0.9, (30, 100)) #USD - Cost for running an ad 60 times
ad_months_per_station = 3 #How long would the campaign run for each year per station
days_per_month = 365.25/12
ad_days_per_station = ad_months_per_station * days_per_month
#Total number of ads per year
num_ads_per_year = num_radio_stations * ad_days_per_station * num_ads_per_day_per_station 
total_ad_cost = cost_per_60_ads * num_ads_per_year #Total cost of airing ads per year, assuming we run each ad 60 times

# Radio edutainment – airtime costs
cost_edutainment_per_minute_per_ad = beta(10 * 1, 10 * 1)
cost_edutainment_per_minute = cost_edutainment_per_minute_per_ad * cost_per_60_ads # I think there might be an error in the orignal computation here. At least, it's not explicitly referencing the 60 ads figure, which seems suspicious
len_edutainment_show = 20 #minutes - Length of edutainment show
cost_per_edutainment_show = cost_edutainment_per_minute * len_edutainment_show #Cost of one edutainment show
edutainment_campaign_months_per_station = 3 #How long would the campaign run for each year per station
edutainment_show_per_week_per_station = 3 #Number of shows per week per station
num_edutainment_shows_per_year = (num_radio_stations *
    edutainment_campaign_months_per_station /
    12 *
    edutainment_show_per_week_per_station *
    52 #Total number of shows per year
    )
total_airtime_cost = num_edutainment_shows_per_year * cost_per_edutainment_show #Total cost of airing shows per year

# Radio advertising – production costs
cost_per_ad_production = ci.log_normal(0.9, [50, 400]) #USD - Cost of production of one ad
num_times_repeat_ad = 50 #Number of times we repeat each advert [Note, didn't we assume 60 above? I'll ignore this]
num_different_ads_per_year = num_ads_per_year / num_times_repeat_ad #How many different ads in total
total_ad_production_cost_per_year = ( #Total production cost
    num_different_ads_per_year *
    cost_per_ad_production) 

# Radio edutainment – production costs
cost_of_edutainment_show = ci.log_normal(0.9, [250 / 2, 2500 * 2]) #USD - Cost of production of one edutainment show
num_times_repeat_edutainment_show = ci.log_normal(0.9, [3,6]) #Average number of times repeat each edutainment show
num_different_edutainment_shows_per_year = ( #How many different shows
    num_edutainment_shows_per_year / num_times_repeat_edutainment_show)
total_edutainment_production_cost_per_year = ( #Total production cost
    num_different_edutainment_shows_per_year *
    cost_of_edutainment_show) 

# Total variable charity costs for production and airtime
total_variable_charity_costs = (
    total_ad_cost + total_airtime_cost +
    total_ad_production_cost_per_year +
    total_edutainment_production_cost_per_year)

# NON-CHARITY COSTS
# Costs to government
cost_to_government = 0 #USD
cost_to_other_charities = 0 #USD
cost_to_other_non_benficaries = 0 #USD
total_non_charity_costs = (cost_to_government + cost_to_other_charities +
    cost_to_other_non_benficaries)


# In[14]:


yearly_scale_of_charity_ops_if_successful = {
    year: 0 if year==1 else 1/cover_time if year <= run_time_successful else 0
    for year in years}
yearly_scale_of_charity_ops = {
    year: ((percent_of_area_if_successful:=yearly_scale_of_charity_ops_if_successful[year]) 
        if year <= run_time_successful 
        else percent_of_area_if_successful * prob_of_success)
    for year in years
}

def charity_cost_in_year(year):
    overhead = (
        overhead_year_1 if year==1
        else overhead_scale if year <= run_time_unsuccessful
        else overhead_scale * prob_of_success if year <= run_time_successful
        else 0)
    variable_costs = total_variable_charity_costs * yearly_scale_of_charity_ops[year] 
    return overhead + variable_costs
    
yearly_charity_cost = {
    year: charity_cost_in_year(year) for year in years
    }

yearly_non_charity_cost = {
    year: total_non_charity_costs * yearly_scale_of_charity_ops[year]
    for year in years
}

# # EFFECTS

# ## REACH OF CAMPAIGN

def num_families_exposed_estimate1():
    avg_audience_size_radio_station = ci.log_normal(0.9, [1*M, 3*M]) #Average audience size of a radio station
    overlap_radio_stations = 20 / 100 #Assume overlap of radio stations so less net audience
    num_listeners_exposed_to_campaign_per_station = (
        avg_audience_size_radio_station
        * (1 - percent_population_rural)
        * overlap_radio_stations)
    num_listeners_exposed_to_campaign = (
        num_radio_stations 
        * num_listeners_exposed_to_campaign_per_station)
    num_listeners_per_household = ci.log_normal(0.9, [2, 4]) 
    return num_listeners_exposed_to_campaign / num_listeners_per_household 


def num_families_exposed_estimate2():
    radio_coverage_in_rural_areas = 78.8 / 100 
    percent_areas_covered_by_campaign = beta(2 * 3, 2 * 1) #Percentage of these areas covered by the campaign
    percent_families_with_radio_who_receive_message = beta(5 * 1, 5 * 1) #Percentage of families with radio who receive the message
    people_exposed_to_campaign = (
        rural_population
        * radio_coverage_in_rural_areas
        * percent_areas_covered_by_campaign
        * percent_families_with_radio_who_receive_message) 
    return people_exposed_to_campaign / avg_household_size #Number of families exposed to the campaign


# Number reached – total per year
num_families_exposed_to_campaign = (
    num_families_exposed_estimate1() + num_families_exposed_estimate2()) / 2 


# ## BEHAVIOUR CHANGE (IN AUDIENCE REACHED)
# Percentage of message-receiving families who send a migrant within 2 years
def percent_families_sending_migrant_within_2_years():
    #Percentage of message-receiving families who send a migrant within 2 years                                                   
    pfsmw2y_estimate = beta(0.3 * 12.3, 0.3 * (100 - 12.3))
    internal_validity_discount = 50 / 100 #Internal validity discount
    external_validity_discount = 30 / 100 #External validity discount
    return pfsmw2y_estimate * (1 - internal_validity_discount) * (1 - external_validity_discount) # the discounted value

number_of_migrants_sent_per_family = 1 
new_migrants_per_year = (
    num_families_exposed_to_campaign
    * number_of_migrants_sent_per_family 
    * percent_families_sending_migrant_within_2_years())

yearly_total_migrants = {
    year: new_migrants_per_year * prob_of_success * yearly_scale_of_charity_ops_if_successful[year]
    for year in years
}


# ## Income & consumption effects on the migrant
# Short-term earning effects on the migrant – initial estimate
def _short_term_earning_effects_on_migrant_initial_estimate():
    #Short-term earning effects on the migrant – initial estimate with mean about 160%
    stea_estimate = ci.log_normal(0.9, (1.1,2.3))
    internal_validity_discount = 50 / 100 #Internal validity discount
    external_validity_discount = 30 / 100 #External validity discount
    return stea_estimate * (1 - internal_validity_discount) * (1 - external_validity_discount) # the discounted value
short_term_earning_effects_on_migrant_initial_estimate = _short_term_earning_effects_on_migrant_initial_estimate()

percent_sent_back_as_remittances = 22 / 100 #Percentage of extra income sent back as remittances [Enter some uncertainty here]
percent_consumed_by_migrant = 90 / 100 #Proportion of extra income that is consumed by migrant [Enter some uncertainty here]
percent_of_increased_consumption = (
    short_term_earning_effects_on_migrant_initial_estimate
    * percent_consumed_by_migrant 
    * (1 - percent_sent_back_as_remittances))
consumption_doublings = log2(1 + percent_of_increased_consumption) # Consumption doublings

# ## Effects on the migrant's family
increased_family_income_due_to_remittences = (percent_sent_back_as_remittances *
  short_term_earning_effects_on_migrant_initial_estimate)
increased_family_income_due_to_higher_local_wages = 2.3 / 100 #Increase in income due to higher local wages
percent_consumed_by_village_dwellers = 90 / 100 #Proportion of extra income that is consumed by village dwellers
def _total_increased_family_consumption():
    income_effect = (increased_family_income_due_to_higher_local_wages +
        increased_family_income_due_to_remittences)
    validity_discount = 40 / 100
    return income_effect * (1 - validity_discount) * percent_consumed_by_village_dwellers
consumption_increase_per_family_member = (_total_increased_family_consumption() /
  (avg_household_size - 1))

# Harms / costs to beneficaries
def _cost_of_one_way_trip_by_bus():
    cost_of_one_way_trip_by_bus_bangaladesh = 5 #USD - Cost of a one-way bus trip in Bangladesh
    rwanda_gdp_per_capita = 822 #USD - GDP per capita - Rwanda - 2021
    bangladesh_gdp_per_capita = 630 #USD - GDP per capita - Bangladesh - 2008
    return (cost_of_one_way_trip_by_bus_bangaladesh 
            * rwanda_gdp_per_capita / bangladesh_gdp_per_capita)
       # Assuming that cost scales roughly proportionately with GDP per capita
cost_of_one_way_trip_by_bus = _cost_of_one_way_trip_by_bus()


# In[21]:


def indexed_migrants_by_year_or_0(year):
    return 0 if year < 1 else yearly_total_migrants[year]


# In[22]:


def total_working_migrants_in(year):
    assert year>0
    workers_new = (indexed_migrants_by_year_or_0(year) *
      (1 - migrate_return_ratios["return without finding job"]))
    workers_1_year = (indexed_migrants_by_year_or_0(year - 1) 
        * (1 - migrate_return_ratios["return after 1 year"] -
           migrate_return_ratios["return without finding job"]))
    # workers_2_years = indexed_migrants_by_year_or_0(year-2) * (1 - migrate_return_ratios["return after 2 years"] - migrate_return_ratios["return after 1 year"] - migrate_return_ratios["return without finding job"])  // this isn't used in the original model, an error I think
    workers_indefinite = (
        # sum the list of all years before the current year
        sum(map(indexed_migrants_by_year_or_0, years[:year-1])) 
        * migrate_return_ratios["stay indefinitely"] 
    )
    return workers_new + workers_1_year + workers_indefinite

yearly_total_working_migrants = {year: total_working_migrants_in(year) for year in years}

def total_returning_migrants_in(year):
    mrr = migrate_return_ratios
    return (
        indexed_migrants_by_year_or_0(year) * mrr["return without finding job"]
        +indexed_migrants_by_year_or_0(year - 1) * mrr["return after 1 year"]
        +indexed_migrants_by_year_or_0(year - 2) * mrr["return after 2 years"])
        
yearly_total_returning_migrants = {
    year: total_returning_migrants_in(year) for year in years
}

yearly_avg_time_migrant_at_destination = {
    year: 1 # TODO placeholder
    for year in years 
}

_yearly_increase_in_consumption = {
    year: (percent_of_increased_consumption *
        (1 + rise_in_effect_each_year) **
        minimum(
            yearly_avg_time_migrant_at_destination[year],
            years_before_plateau
          ))
    for year in years
}

yearly_migrant_consumption_doublings = {
    year: yearly_total_working_migrants[year]*log2(1+_yearly_increase_in_consumption[year])
    for year in years
}

def _family_increase_in_consumption(year):
    return (consumption_increase_per_family_member * 
            (1 + rise_in_effect_each_year)**(minimum(year, years_before_plateau)))
yearly_family_consumption_doublings = {
    year: (yearly_total_working_migrants[year] 
           * (avg_household_size-1) 
           * log2(1+_family_increase_in_consumption(year)))
    for year in years
}

def yearly_outward_cost_in_doublings(year): 
    decrease_in_consumption = cost_of_one_way_trip_by_bus / avg_consumption 
    total_migrants = yearly_total_migrants[year]
    return total_migrants * log2(1-decrease_in_consumption)

def yearly_return_cost_in_doublings(year):
    adjusted_avg_consumption = avg_consumption * (1 + _yearly_increase_in_consumption[year])
    decrease_in_consumption = cost_of_one_way_trip_by_bus / adjusted_avg_consumption
    total_migrants = yearly_total_returning_migrants[year]
    return total_migrants * log2(1-decrease_in_consumption)

def yearly_total_consumption_doublings(year):
    return (
        yearly_migrant_consumption_doublings[year] +
        yearly_family_consumption_doublings[year] +
        yearly_outward_cost_in_doublings(year) +
        yearly_return_cost_in_doublings(year)
    )

def dict_from_fun(f, keys):
    return {key:f(key) for key in keys}

def npv(discount_rate: float, 
        cashflows: Mapping[int, float], 
        reference_year: int = 1
       ) -> float: 
    result = 0
    for years_from_now, cashflow in cashflows.items():
        result += cashflow / (1+discount_rate)**(years_from_now - reference_year)
    return result

total_charity_cost = npv(discount_rate, yearly_charity_cost)
total_non_charity_cost = npv(discount_rate, yearly_non_charity_cost)
total_consumption_doublings = npv(discount_rate, 
                                  dict_from_fun(yearly_total_consumption_doublings,years))
total_DALYs = total_consumption_doublings * consumption_doubling_to_DALY_ratio
total_cost = total_non_charity_cost + total_charity_cost

def define_deterministic(context, names):
    return {name: nba_pymc.deterministic(context[name], name=name) for name in names}

to_show = define_deterministic(locals(), [
  'total_charity_cost',
  'total_cost',
  'total_consumption_doublings',
  'total_DALYs'])

def samples_path(n_samples: int) -> str:
    return f"rural_to_urban.{n_samples}.nc"

def plot_dists(samples, vars_to_show):
    import seaborn
    for name in vars_to_show:
        print(name)
        seaborn.kdeplot(getattr(samples, name).values.ravel())

def run(n_samples = 100000):
    print("sampling...", end='')
    samples = nba_pymc.sample_prior_predictive(samples=n_samples, var_names=list(to_show.keys())).prior
    print(" done.")

    samples.to_netcdf(samples_path(n_samples))

    plot_dists(samples, to_show)
    import matplotlib.pyplot as plt
    plt.show()

if __name__=="__main__":
    run()

