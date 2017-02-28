#! /bin/bash
set -e
javac -cp "racetrack.jar" QLearning.java
java  -Djna.library.path=/usr/local/lib -cp "racetrack.jar" QLearning
