import numpy as np
import fuzzy as fuzz
import control as ctrl

# Antecedentes
universo_velocidad = np.arange(0, 1001, 1)
universo_angulo = np.arange(-10, 11, 0.1)

# Consecuentes
universo_posicion = np.arange(0, 11, 1)

velocidad = ctrl.Antecedent(universo_velocidad, "velocidad")
angulo = ctrl.Antecedent(universo_angulo, "angulo")
posicion = ctrl.Consequent(universo_posicion, "posicion")

# Velocidad
velocidad["alta"] = fuzz.trapmf(universo_velocidad, [600, 800, 1000, 1000])
velocidad["media"] = fuzz.trimf(universo_velocidad, [300, 500, 700])
velocidad["baja"] = fuzz.trapmf(universo_velocidad, [0, 0, 250, 400])
velocidad.view()

# Angulo
angulo["ascenso"] = fuzz.trapmf(universo_angulo, [2, 5, 10, 10])
angulo["nivel"] = fuzz.trimf(universo_angulo, [-3, 0, 3])
angulo["descenso"] = fuzz.trapmf(universo_angulo, [-10, -10, -5, -2])
angulo.view()

# posicion
posicion["muy alta"] = fuzz.trapmf(universo_posicion, [8, 9, 10, 10])
posicion["alta"] = fuzz.trimf(universo_posicion, [7, 8, 9])
posicion["media"] = fuzz.trimf(universo_posicion, [6, 7, 8])
posicion["baja"] = fuzz.trimf(universo_posicion, [4, 5, 6])
posicion["muy baja"] = fuzz.trapmf(universo_posicion, [0, 0, 3, 4])
posicion.view()

regla_muy_alta = ctrl.Rule(velocidad["alta"] & angulo["ascenso"], (posicion, "muy alta"))
regla_alta = ctrl.Rule((velocidad["media"] & angulo["ascenso"]) | (velocidad["alta"] & angulo["nivel"]), (posicion, "alta"))
regla_media = ctrl.Rule((velocidad["baja"] & angulo["ascenso"]) | (velocidad["media"] & angulo["nivel"]) | (velocidad["baja"] & angulo["descenso"]), (posicion, "media"))
regla_baja = ctrl.Rule((velocidad["baja"] & angulo["nivel"]) | (velocidad["media"] & angulo["descenso"]), (posicion, "baja"))
regla_muy_baja = ctrl.Rule(velocidad["alta"] & angulo["descenso"], (posicion, "muy baja"))

control_vuelo = ctrl.ControlSystem([regla_muy_alta, regla_alta, regla_media, regla_baja, regla_muy_baja])

posicion_timon = ctrl.ControlSystemSimulation(control_vuelo)
posicion_timon.input["velocidad"] = 200
posicion_timon.input["angulo"] = 10
posicion_timon.compute()

velocidad.view(sim=posicion_timon)
angulo.view(sim=posicion_timon)
posicion.view(sim=posicion_timon)

print(f"Posición del timón: {posicion_timon.output['posicion']}")