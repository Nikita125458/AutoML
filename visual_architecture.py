import netron
import time

import netron

import subprocess

subprocess.Popen(["netron", "models/neural/bank/architecture1.keras", "--port", "8081"])
subprocess.Popen(["netron", "models/neural/bank/architecture2.keras", "--port", "8082"])

subprocess.Popen(["netron", "models/neural/churn/architecture1.keras", "--port", "8083"])
subprocess.Popen(["netron", "models/neural/churn/architecture2.keras", "--port", "8084"])

subprocess.Popen(["netron", "models/neural/housing/architecture1.keras", "--port", "8085"])
subprocess.Popen(["netron", "models/neural/housing/architecture2.keras", "--port", "8086"])
