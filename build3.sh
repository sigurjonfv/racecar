#! /bin/bash
set -e
javac -cp "racetrack.jar:fannj-0.7.jar:jna-4.2.2.jar:commons-io-2.5.jar" QLearningNN.java
java  -Djna.library.path=/usr/local/lib -cp "racetrack.jar:fannj-0.7.jar:jna-4.2.2.jar:commons-io-2.5.jar" QLearningNN
