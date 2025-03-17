import sys
import time
import subprocess
import pandas as pd
import numpy as np
import psutil
import os
from datetime import datetime

class CPUWatcher:
    def __init__(self, process_pid=None, max_rows: int = 10000, base_folder=None):
        self.process_pid = process_pid
        self.process = psutil.Process(process_pid) if process_pid is not None else None
        self.max_rows = max_rows
        self.general_data = {
            "CPU_usage": [],
            "Memory_info": [],
            "DiskUsage": [],
            "nWattsSupply": [],
            "Moment": [],
        }
        self.process_data = {
            "Process_CPU": [],
            "Affined_Cores": [],
            "Affined_Temperatures": [],
            "Timestamp": [],
        }
        self.n_saves = 0

        # Si no se proporciona un directorio base, se usa "outputs/cpu"
        if base_folder is None:
            base_folder = os.path.join("outputs", "cpu")
        self.base_folder = base_folder
        # Crear subcarpetas: threads, cores, general y process
        for subfolder in ["threads", "cores", "general", "process"]:
            dir_path = os.path.join(self.base_folder, subfolder)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        self.threads_dir = os.path.join(self.base_folder, "threads")
        self.cores_dir = os.path.join(self.base_folder, "cores")
        self.general_dir = os.path.join(self.base_folder, "general")
        self.process_dir = os.path.join(self.base_folder, "process")

    def get_cpu_core_data(self):
        """
        Retorna dos DataFrames:
          - df_fisico: datos agregados por núcleo físico.
          - df_logico: datos individuales por núcleo lógico (hilo).
        Se asigna la temperatura de cada hilo según su núcleo.
        Si no se encuentran sensores con etiqueta "Core", se usa el promedio de "Tdie".
        """
        num_cores_fisicos = psutil.cpu_count(logical=False)
        num_cores_logicos = psutil.cpu_count(logical=True)
        cpu_percentages = psutil.cpu_percent(interval=1, percpu=True)
        cpu_frequencies = psutil.cpu_freq(percpu=True)

        temps_per_core = {}
        try:
            temps_dict = psutil.sensors_temperatures()
            if temps_dict:
                found_core = False
                for sensor_group in temps_dict.values():
                    for sensor in sensor_group:
                        if sensor.label and "Core" in sensor.label:
                            found_core = True
                            try:
                                parts = sensor.label.split()
                                if len(parts) >= 2 and parts[0] == "Core":
                                    core_id = int(parts[1])
                                    temps_per_core[core_id] = sensor.current
                            except Exception:
                                continue
                if not found_core:
                    tdie_temps = []
                    for sensor_group in temps_dict.values():
                        for sensor in sensor_group:
                            if sensor.label and "Tdie" in sensor.label:
                                tdie_temps.append(sensor.current)
                    if tdie_temps:
                        avg_temp = sum(tdie_temps) / len(tdie_temps)
                        for core_id in range(num_cores_fisicos):
                            temps_per_core[core_id] = avg_temp
        except Exception:
            temps_per_core = {}

        threads_per_physical = num_cores_logicos // num_cores_fisicos if num_cores_fisicos else 1

        threads_data = []
        for i in range(num_cores_logicos):
            freq = cpu_frequencies[i].current if cpu_frequencies and i < len(cpu_frequencies) else None
            physical_index = i // threads_per_physical if threads_per_physical else i
            temp = temps_per_core.get(physical_index, None)
            threads_data.append({
                "Núcleo lógico": i,
                "Uso (%)": cpu_percentages[i],
                "Frecuencia (MHz)": freq,
                "Temperatura (°C)": temp,
            })
        df_logico = pd.DataFrame(threads_data)

        groups = np.array_split(df_logico, num_cores_fisicos)
        physical_data = []
        for idx, group in enumerate(groups):
            uso_prom = group["Uso (%)"].mean()
            freq_prom = group["Frecuencia (MHz)"].mean() if group["Frecuencia (MHz)"].notna().any() else None
            temp_prom = group["Temperatura (°C)"].mean() if group["Temperatura (°C)"].notna().any() else None
            physical_data.append({
                "Núcleo físico": idx,
                "Uso (%)": uso_prom,
                "Frecuencia (MHz)": freq_prom,
                "Temperatura (°C)": temp_prom,
                "Hilos asociados": len(group)
            })
        df_fisico = pd.DataFrame(physical_data)
        return df_fisico, df_logico

    def read_power_supply(self):
        path = "/sys/class/power_supply/BAT0/"
        try:
            with open(path + "power_now", "r") as power_file:
                power_now = float(power_file.read().strip())
            return power_now / 1e6
        except FileNotFoundError:
            return -1

    def record_general_data(self):
        if self.process is not None:
            cpu_usage = self.process.cpu_percent(interval=None)
        else:
            cpu_usage = psutil.cpu_percent(interval=1)
        self.general_data["CPU_usage"].append(cpu_usage)
        self.general_data["Memory_info"].append(psutil.virtual_memory())
        self.general_data["DiskUsage"].append(psutil.disk_usage('/'))
        self.general_data["nWattsSupply"].append(self.read_power_supply())
        self.general_data["Moment"].append(str(datetime.now()))

    def record_cpu_data(self):
        df_fisico, df_logico = self.get_cpu_core_data()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_logico.to_csv(os.path.join(self.threads_dir, f'{timestamp_str}.csv'), index=False)
        df_fisico.to_csv(os.path.join(self.cores_dir, f'{timestamp_str}.csv'), index=False)

    def record(self):
        self.record_general_data()
        self.record_cpu_data()
        if len(self.general_data["CPU_usage"]) > self.max_rows:
            self.save()

    def record_process_cpu_data(self):
        if self.process is not None:
            process_cpu = self.process.cpu_percent(interval=0.1)
            affinity = self.process.cpu_affinity()
            global_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
            cores_usage = {core: global_cpu[core] for core in affinity}

            temps_per_core = {}
            try:
                temps_dict = psutil.sensors_temperatures()
                if temps_dict:
                    found_core = False
                    for sensor_group in temps_dict.values():
                        for sensor in sensor_group:
                            if sensor.label and "Core" in sensor.label:
                                found_core = True
                                try:
                                    parts = sensor.label.split()
                                    if len(parts) >= 2 and parts[0] == "Core":
                                        core_id = int(parts[1])
                                        temps_per_core[core_id] = sensor.current
                                except Exception:
                                    continue
                    if not found_core:
                        tdie_temps = []
                        for sensor_group in temps_dict.values():
                            for sensor in sensor_group:
                                if sensor.label and "Tdie" in sensor.label:
                                    tdie_temps.append(sensor.current)
                        if tdie_temps:
                            avg_temp = sum(tdie_temps) / len(tdie_temps)
                            for core_id in affinity:
                                temps_per_core[core_id] = avg_temp
            except Exception:
                temps_per_core = {}

            cores_temperatures = {core: temps_per_core.get(core, None) for core in affinity}
            timestamp = str(datetime.now())
            self.process_data["Process_CPU"].append(process_cpu)
            self.process_data["Affined_Cores"].append(cores_usage)
            self.process_data["Affined_Temperatures"].append(cores_temperatures)
            self.process_data["Timestamp"].append(timestamp)

    def save(self):
        df = pd.DataFrame(self.general_data)
        df.to_csv(os.path.join(self.general_dir, f'general_part_{self.n_saves}.csv'), index=False)
        self.n_saves += 1

    def save_process_data(self):
        pd.DataFrame(self.process_data).to_csv(os.path.join(self.process_dir, 'process_cpu_metrics.csv'), index=False)

class GPUWatcher:
    def __init__(self, gpu_index: int = 1, max_rows: int = 10000, pids: list = [], base_folder=None):
        if base_folder is None:
            base_folder = os.path.join("outputs", "gpu")
        self.base_folder = base_folder
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
        self.gpu_index = gpu_index
        self.max_rows = max_rows
        self.n_saves = 0
        self.headers = [
            "index",
            "timestamp",
            "name",
            "temperature.gpu",
            "utilization.gpu",
            "utilization.memory",
            "memory.total",
            "memory.used",
            "memory.free",
            "power.draw",
            "fan.speed",
            "pstate",
            "clocks.current.graphics",
            "clocks.current.memory"
        ]
        self.general_command = [
            "nvidia-smi",
            "-i", str(self.gpu_index),
            "--query-gpu=" + ",".join(self.headers),
            "--format=csv,noheader,nounits"
        ]
        self.pid_command = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits"
        ]
        self.general_data = []
        self.watching_pids = {}
        if len(pids) > 0:
            for pid in pids:
                pid_str = str(pid)
                self.watching_pids[pid_str] = []
                pid_folder = os.path.join(self.base_folder, pid_str)
                if not os.path.exists(pid_folder):
                    os.makedirs(pid_folder)
        self.general_folder = self.base_folder

    def record(self):
        resultado = subprocess.run(self.general_command, capture_output=True, text=True)
        lineas = resultado.stdout.strip().split("\n")
        for linea in lineas:
            fila = [valor.strip() for valor in linea.split(",")]
            fila.insert(0, str(datetime.now()))
            self.general_data.append(fila)
        self.meassure_pids()
        if len(self.general_data) > self.max_rows:
            self.save()

    def save(self):
        filename = os.path.join(self.general_folder, f'gpu_metrics_part_{self.n_saves}.csv')
        pd.DataFrame(self.general_data, columns=["timestamp"] + self.headers).to_csv(filename, index=False)
        if self.watching_pids:
            for pid, data in self.watching_pids.items():
                pid_folder = os.path.join(self.base_folder, pid)
                filename_pid = os.path.join(pid_folder, f'gpu_pid_metrics_part_{self.n_saves}.csv')
                pd.DataFrame(data, columns=["pid", "process_name", "used_gpu_memory"]).to_csv(filename_pid, index=False)
                self.watching_pids[pid].clear()
        self.n_saves += 1
        self.general_data.clear()

    def meassure_pids(self):
        resultado = subprocess.run(self.pid_command, capture_output=True, text=True)
        lineas = resultado.stdout.strip().split("\n")
        for linea in lineas:
            columnas = [col.strip() for col in linea.split(",")]
            if columnas and columnas[0] in self.watching_pids.keys():
                self.watching_pids[columnas[0]].append(columnas)

class Watcher:
    @staticmethod
    def measure_energy_consumption(timelapse: int = 1):
        env = os.environ.copy()
        cuda_device = 2
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        command = [            
            "python3", "distill.py",
            "--dataset=CIFAR100",
            "--ipc=10",
            "--syn_steps=20",
            "--expert_epochs=2",
            "--max_start_epoch=20",
            "--zca",
            "--lr_img=1000",
            "--lr_lr=1e-05",
            "--lr_teacher=0.01",
            "--buffer_path=cifar100/expertdb",
            "--data_path=cifar100/images_db"
        ]

        start_time = time.time()
        proc = subprocess.Popen(command, env=env)
        process_id = proc.pid
        print(f"\n\n\n process id: {process_id}\n\n\n")
        # Crear la carpeta base para este proceso: outputs/<process_id>/ y dentro cpu/ y gpu/
        base_folder = os.path.join("outputs", str(process_id))
        cpu_base_folder = os.path.join(base_folder, "cpu")
        gpu_base_folder = os.path.join(base_folder, "gpu")
        for folder in [base_folder, cpu_base_folder, gpu_base_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        gpu_watcher = GPUWatcher(gpu_index=cuda_device, pids=[proc.pid], base_folder=gpu_base_folder)
        cpu_watcher = CPUWatcher(process_pid=proc.pid, base_folder=cpu_base_folder)

        while proc.poll() is None:
            gpu_watcher.record()
            cpu_watcher.record()
            cpu_watcher.record_process_cpu_data()
            time.sleep(timelapse)

        end_time = time.time()
        total_execution_time = end_time - start_time

        gpu_watcher.save()
        cpu_watcher.save()
        cpu_watcher.save_process_data()

        output_path = os.path.join(base_folder, "execution_time_total.txt")
        with open(output_path, "w") as f:
            f.write(f"Tiempo total de ejecución: {total_execution_time:.2f} segundos\n")

        print(f"Tiempo total de ejecución: {total_execution_time:.2f} segundos")
        print(f"El tiempo de ejecución se ha guardado en: {output_path}")
        return proc.returncode

if __name__ == "__main__":
    print("Uso: python3 meassurator.py [timelapse]")
    timelapse = int(sys.argv[1])
    Watcher.measure_energy_consumption(timelapse)
