import java.util.Scanner;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Vector;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.ArrayList;

class Point {
    final int x;
    final int y;
    Point(int x, int y) { this.x = x; this.y = y; }
}

public class ValueIteration {
    private char[][] racetrack;
    private int imax = 0;
    private int jmax = 0;
    private final int VELOCITY_MAX = 4;
    private final int VELOCITY_MIN = -4;
    private final int VELOCITY_OFFSET = 4;
    private final int ACCEL_MAX = 1;
    private final int ACCEL_MIN = -1;
    private final double GAMMA = 0.9;
    private final int REW_MOVE = -1;
    private final int REW_OOB = -5;
    private final double DELTA = 0.000000000000001;
    private final int NEG_INF = (1<<31);

    /*Set this value to true for visual representation of the policy from every single starting point.
      Set to false for running a number of simulations from a random starting point and outputting
      average, best and worst scores.
    */
    private final boolean VISUALIZE = true;
    private final int NO_SIM = 100000; //Number of simulations to run


    private double[][][][] valueArray;
    private ArrayList<Point> startingPositions;
    private Random r;
    private int bestScore;
    private int worstScore;

    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_BLACK = "\u001B[30m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_YELLOW = "\u001B[33m";
    public static final String ANSI_BLUE = "\u001B[34m";
    public static final String ANSI_PURPLE = "\u001B[35m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_WHITE = "\u001B[37m";

    public ValueIteration(String filename) {
        r = new Random();
        try {
            Scanner in = new Scanner(new File(filename));
            this.jmax = in.nextInt();
            this.imax = in.nextInt();
            System.out.println(jmax + " " + imax);
            int v = VELOCITY_MAX - VELOCITY_MIN;
            this.racetrack = new char[imax][jmax];
            in.useDelimiter("\n"); // To read in characters
            for (int i = 0; i < imax; i++) {
                String line = in.next();
                if(line.trim().isEmpty())
                {
                    i--;
                    continue;
                }
                for (int j = 0; j < jmax; j++) {
                    racetrack[i][j] = (char) line.charAt(j);
                }
            }
        } catch (FileNotFoundException e) {
            System.out.printf("The file %s was not found or could not be opened\n", filename);
            return;
        }
        initializeValues();
        printTrack(-1, -1);
        findOptimal();
        if(VISUALIZE)
            episodeVisualization();
        else
            System.out.printf("Average score over %d simulations with random starting points was %f\nBest score was %d and worst score was %d.\n", NO_SIM, avgSimulation(NO_SIM), bestScore, worstScore);
    }

    private void initializeValues(){
        this.worstScore = 0;
        this.bestScore = NEG_INF;
        this.startingPositions = new ArrayList<Point>();
        for(int i = 0; i < imax; i++){
            for(int j = 0; j < jmax; j++){
                if(racetrack[i][j] == 'S'){
                    startingPositions.add(new Point(i,j));
                }
            }
        }
    }


    private double max(double a, double b) {
        return a < b ? b : a;
    }

    private double abs(double a) {
        return a < 0 ? -a : a;
    }

    private boolean isFinishline(int i, int j) {
        return racetrack[i][j] == 'F';
    }

    private boolean isOutOfBounds(int i, int j) {
        if (i > imax - 1 || i < 0)
            return true;
        if (j > jmax - 1 || j < 0)
            return true;
        if (racetrack[i][j] == 'X')
            return true;
        return false;
    }

    private boolean isValidSpeed(int vi, int vj) {
        return !(vj > VELOCITY_MAX || vj < VELOCITY_MIN || vi > VELOCITY_MAX || vi < VELOCITY_MIN);
    }

    private double getValue(double [][][][] values, int i, int j, int vi, int vj) {
        return values[i][j][vi + VELOCITY_OFFSET][vj + VELOCITY_OFFSET];
    }

    //Prints the track in pretty colors
    private void printTrack(int cari, int carj) {
        for (int i = 0; i < this.imax; i++) {
            for (int j = 0; j < this.jmax; j++) {
                if (cari == i && carj == j) {
                    System.out.print(ANSI_RED + 'C' + ANSI_RESET);
                } else if (racetrack[i][j] == 'X') {
                    System.out.print(ANSI_GREEN + 'X' + ANSI_RESET);
                } else if (racetrack[i][j] == 'O') {
                    System.out.print(ANSI_BLACK + 'O' + ANSI_RESET);
                } else if (racetrack[i][j] == 'F') {
                    System.out.print(ANSI_BLUE + 'F' + ANSI_RESET);
                } else if (racetrack[i][j] == 'S') {
                    System.out.print(ANSI_YELLOW + 'S' + ANSI_RESET);
                }
            }
            System.out.println();
        }
    }

    //Simulates the policy for a given start point
    private int simulate(int starti, int startj) {
        int nexti = starti, nextj = startj, nextvi = 0, nextvj = 0;
        int score = 0;
        int offTrack = 0;
        boolean finished = false;
        for (int i = 0; i < 1000; i++) {
            int bestai = 0, bestaj = 0;
            double best = Double.NEGATIVE_INFINITY;
            for (int ai = ACCEL_MIN; ai <= ACCEL_MAX; ai++) {
                for (int aj = ACCEL_MIN; aj <= ACCEL_MAX; aj++) {
                    if (isOutOfBounds(nexti + nextvi + ai, nextj + nextvj + aj))
                        continue;
                    if (!isValidSpeed(nextvi + ai, nextvj + aj)) {
                        continue;
                    }
                    double moveValue = getValue(valueArray,
                        nexti + nextvi + ai,
                        nextj + nextvj + aj,
                        nextvi + ai,
                        nextvj + aj);
                    if (moveValue > best) {
                        best = moveValue;
                        bestai = ai;
                        bestaj = aj;
                    }
                }
            }
            if(VISUALIZE)
                System.out.printf("Best line to take from (%d, %d, %d, %d) is (%d, %d) with a value of %f\n",
                    nexti, nextj, nextvi, nextvj, bestai, bestaj, best);
            boolean slidUp = false, slidRight = false;
            double chance = r.nextDouble();
            if (chance > 0.75) { // Up slide
                if(VISUALIZE)
                    System.out.printf("** Car slides up!\n");
                nexti -= 1;
                slidUp = true;
            } else if (chance > 0.5) { // Right slide
                if(VISUALIZE)
                    System.out.printf("** Car slides right!\n");
                nextj += 1;
                slidRight = true;
            }
            nextvi += bestai;
            nextvj += bestaj;
            if (isOutOfBounds(nexti + nextvi, nextj + nextvj)) {
                score += REW_OOB + REW_MOVE;
                nextvi = 0;
                nextvj = 0;
                //If it went out of bounds we have to undo the slide
                if(slidUp)
                    nexti += 1;
                if(slidRight)
                    nextj -= 1;
                offTrack++;
                if(VISUALIZE)
                    System.out.printf("** Car has gone off the track!\n");
            } else {
                nexti += nextvi;
                nextj += nextvj;
                score += REW_MOVE;
            }
            if(VISUALIZE){
                System.out.printf("Car is now at (%d, %d, %d, %d)\n",
                    nexti, nextj, nextvi, nextvj);
                printTrack(nexti, nextj);
            }
            if (isFinishline(nexti, nextj)) {
                if(VISUALIZE)
                    System.out.printf("** Car has finished the track with a score of %d, it went off track %d time(s)\n",
                        score, offTrack);
                return score;
            }
            if(VISUALIZE){
                try {
                    TimeUnit.MILLISECONDS.sleep(100);
                } catch(InterruptedException ex) {
                    Thread.currentThread().interrupt();
                }
            }
        }
        return -5000; //If it doesn't complete in 1000 moves then it gets a score of -5000
    }

    //Finds an (hopefully) optimal policy
    private void findOptimal() {
        int v = VELOCITY_MAX - VELOCITY_MIN + 1;
        double [][][][] start = new double[imax][jmax][v][v];
        double improvement = 1;
        int iteration = 0;
        long startTime = System.currentTimeMillis();
        while (improvement > DELTA) {
            iteration++;
            improvement = valueIteration(start);
            System.out.printf("Learning iteration %d gives improvement %.15f\n",
                iteration, improvement);
        }
        long endTime = System.currentTimeMillis();
        System.out.printf("Time spent learning policy %.2fs\n", (endTime - startTime) / 1000.0);
        this.valueArray = start;
    }


    private void episodeVisualization() {
        try {
            TimeUnit.SECONDS.sleep(3);
        } catch(InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
        for(Point p : startingPositions){
            printTrack(p.x, p.y);
            try {
                TimeUnit.SECONDS.sleep(1);
            } catch(InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
            simulate(p.x, p.y);
            try {
                TimeUnit.SECONDS.sleep(2);
            } catch(InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private double avgSimulation(int n) {
        int total = 0;
        for(int i = 0; i < n; i++){
            int score = randomSimulation();
            total += score;
            this.bestScore = Math.max(this.bestScore, score);
            this.worstScore = Math.min(this.worstScore, score);
        }
        return total / (double) n;
    }

    private int randomSimulation() {
            int ind = r.nextInt(startingPositions.size());
            return simulate(startingPositions.get(ind).x, startingPositions.get(ind).y);
    }

    private double stateValue(double [][][][] start, int i, int j, int vi, int vj) {
        if (i > imax - 1 || i < 0 || j > jmax - 1 || j < 0) {
            return Double.NEGATIVE_INFINITY;
        }
        double total = 0;
        double val = getValue(start, i, j, vi, vj);
        total = 0.5 * (GAMMA * val + REW_MOVE);
        if (isFinishline(i, j)) {
            total = 0;
        }
        if (isOutOfBounds(i, j)) {
            total = 0.5 * (REW_OOB + REW_MOVE);
        }
        if (isOutOfBounds(i - 1, j)) {
            total += 0.25 * (GAMMA * getValue(start, i, j, 0, 0) + REW_OOB + REW_MOVE);
        } else {
            total += 0.25 * (GAMMA * getValue(start, i - 1, j, vi, vj) + REW_MOVE);
        }
        if (isOutOfBounds(i, j + 1)) {
            total += 0.25 * (GAMMA * getValue(start, i, j, 0, 0) + REW_OOB + REW_MOVE);
        } else {
            total += 0.25 * (GAMMA * getValue(start, i, j + 1, vi, vj) + REW_MOVE);
        }
        return total;
    }

    public double valueIteration(double [][][][] start) {
        double improvement = 0;
        double [][][][] next = start.clone();
        for (int i = 0; i < imax; i++) {
            for (int j = 0; j < jmax; j++) {
                if(isFinishline(i,j) && !isOutOfBounds(i-1, j))
                    continue;
                for (int vi = VELOCITY_MIN; vi <= VELOCITY_MAX; vi++) {
                    for (int vj = VELOCITY_MIN; vj <= VELOCITY_MAX; vj++) {
                        // Best result of any accelerations
                        double best = Double.NEGATIVE_INFINITY;
                        for (int ai = ACCEL_MIN; ai <= ACCEL_MAX; ai++) {
                            for (int aj = ACCEL_MIN; aj <= ACCEL_MAX; aj++) {
                                int newvj = vj + aj;
                                int newvi = vi + ai;
                                int newj = j + newvj;
                                int newi = i + newvi;
                                if (!isValidSpeed(newvi, newvj)) {
                                    continue;
                                } else {
                                    double val = stateValue(start, newi, newj, newvi, newvj);
                                    best = max(best, val);
                                }
                            }
                        }
                        improvement = max(improvement, abs(getValue(start, i, j, vi, vj) - best));
                        next[i][j][vi + VELOCITY_OFFSET][vj + VELOCITY_OFFSET] = best;
                    }
                }
            }
        }
        // Replace old array with the new one and return the improvement we made
        start = next;
        return improvement;
    }

    public static void main(String[] args) throws Exception {
        System.out.println(args[0]);
        new ValueIteration(args[0]);
    }
}
