# Final Navigation

Stack de navegacion autonoma para ROS 2 Jazzy. Utiliza el mundo `depot`, un
costmap propio, RRT* como planificador global, DWA como controlador local y un
modo reactivo de seguimiento de borde para obstaculos que no estan en el mapa.

## Dependencias

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic y `ros_gz`
- RViz2
- Python 3.12
- `colcon`

Instalacion de dependencias ROS y Python en Ubuntu:

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-xacro \
  ros-jazzy-robot-state-publisher \
  python3-colcon-common-extensions \
  python3-yaml \
  python3-numpy \
  python3-matplotlib
```

Las dependencias Python tambien estan documentadas en `requirements.txt`.

## Estructura

```text
ros2_packages/       # paquetes ROS 2 de simulacion, robot, AMCL y EKF
final_navigation/    # stack propio de navegacion autonoma
data/                # rosbag/MCAP de pruebas
```

## Compilacion

Desde la carpeta principal del proyecto:

```bash
cd /home/fernando-quintero/Descargas/navegacion
source /opt/ros/jazzy/setup.bash
colcon build --packages-select \
  amcl_launch ekf_localization el7009_diff_drive_robot final_navigation
source install/setup.bash
```

## Ejecucion

### Terminal 1: simulacion

```bash
cd /home/fernando-quintero/Descargas/navegacion
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch el7009_diff_drive_robot robot.launch.py
```

Este comando abre Gazebo con el mundo `depot`, el robot diferencial y RViz.

### Terminal 2: stack de navegacion

Ejecutar cuando Gazebo ya este cargado:

```bash
cd /home/fernando-quintero/Descargas/navegacion
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch final_navigation final_navigation.launch.py \
  launch_sim:=false use_sim_time:=true \
  run_rrt_star:=true run_dwa:=true \
  run_recorder:=true run_manager:=true
```

### Terminal 3: enviar coordenadas

```bash
cd /home/fernando-quintero/Descargas/navegacion
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run final_navigation interactive_goal_sender
```

Escribir cada objetivo con el formato:

```text
x y yaw
```

Ejemplos:

```text
8.0 0.0 0.0
14.0 2.0 1.57
-5.0 -5.0 3.14
```

Las coordenadas `x` e `y` estan expresadas en metros en el frame `map`. El
angulo `yaw` se escribe en radianes: `0` apunta hacia `+x`, `1.57` hacia `+y`,
`3.14` hacia `-x` y `-1.57` hacia `-y`.

La interfaz tambien acepta:

```text
simple   # objetivo (8.0, 0.0, 0.0)
hard     # objetivo (14.0, 2.0, 0.0)
invalid  # objetivo fuera del mapa
stop     # publica velocidad cero
q        # salir
```

## Resultados

Las trayectorias, metricas y figuras se guardan en:

```text
final_navigation/results/
final_navigation/results/figures/
```

Para detener el proyecto, presionar `Ctrl+C` en las terminales abiertas.
