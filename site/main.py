from flask import Flask, render_template, request, redirect, url_for
from optimization import *
from site_functions import *
import time

app = Flask(__name__)

@app.route('/Send_House',methods=['POST','GET'])
def sendhouse():
    house_name=None
    data_machine = None

    if "house_name" in request.form:
        house_name = request.form["house_name"]

    if house_name:
        data_machine = get_machines_of_house(house_name)

    return render_template('optimization.html',data=data_machine,house_name=house_name)

@app.route('/Optimize_Me',methods=['POST','GET'])
def optimizeme():
    response_dict = dict(request.form)
    house_id = response_dict["house_name"]

    modif_house(response_dict)

    solutions = optimization_main()
    house_planning = None

    for key in solutions:
        if house_id in solutions[key][0]:
            house_planning = solutions[key][0][house_id]

    path_plan_img = plot_planning(house_planning,house_id)

    return render_template('optimization.html',img_path=path_plan_img)

@app.route('/Admin', methods=['POST','GET'])
def admin():
    MOY_PS_DAY = 512294.7158481944
    
    start = time.time()
    solutions = optimization_main()
    end = time.time()
    time_elapsed = round(end-start,4)

    All_PL_power = {}
    PS_day_power = []
    for i in range(48):
        PS_day_power.append(0)

    for key in solutions:
        PL_day_power = solutions[key][1]
        All_PL_power[key] = max(PL_day_power)
        for i in range(48):
            PS_day_power[i] += PL_day_power[i]

    ind = PS_day_power.index(max(PS_day_power))
    PS_day_power[ind] = PS_day_power[ind-1]

    (sum_init,sum_recuit) = get_init_cost_random()

    sum_pl = sum([ All_PL_power[key] for key in All_PL_power ])

    path = "static/images/PS_day_conso.png"
    plt.figure(figsize=(13,8))
    plt.title("Consomation au Poste Source au cours de la journée")
   
    ax= plt.subplot()
    ax.plot(PS_day_power)
    plt.savefig(path)
    plt.close()

    return render_template('admin_optimization.html',data=[round(sum_pl),round(sum_init),round(100-(round(sum_pl)*100/round(sum_init)))],time_elapsed=time_elapsed,path=path)

@app.route('/')
def root():
    return render_template('optimization.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8088, debug=True)