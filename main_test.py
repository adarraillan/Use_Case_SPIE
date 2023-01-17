import class_individual

HC = {"start":0,"end":15}

day_consumption = []
for i in range(48):
    day_consumption.append(0)

plannings = [
    {'LV':[12,13,14,15],
    'LL':[0,1,46,47]
    },
    {
    'CE_1':[0,2,8,15],
    'CE_2':[1,2,14,15]
    }
]

matrice_type_puissance_par_demie_heure = {'LV':0.65, 'LL':1, 'SL':0.125, 'TV':0.1, 'FG_1':0.1, 'CE_1': 0.18, 'CG': 0.1, 
                                                'FO': 1.6, 'PL': 1.2, 'FG_2': 0.3, 'CE_2': 0.25}


def calc_consumption_from_planing(plan,mat,day_consumption):
    for house in plan:
        for machine in house.keys():
            for time_slot in house[machine]:
                day_consumption[time_slot] += mat[machine]
    return day_consumption


day_consumption = calc_consumption_from_planing(plannings,matrice_type_puissance_par_demie_heure,day_consumption)

individual = class_individual.Individual(HC,plannings,day_consumption)

print("Before")
print(individual.plannings)
print(individual.day_consumption)

individual.mutate()

print("After")
print(individual.plannings)
print(individual.day_consumption)


for i in range(100000):
    if i%100 == 0:
        print("Done",i)
    individual.mutate()