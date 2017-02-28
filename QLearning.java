import java.util.Scanner;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Vector;
import java.util.Random;
import java.util.concurrent.TimeUnit;

//import com.googlecode.fannj.Fann;


public class QLearning {
    private char[][] racetrack;
    private int imax = 0;
    private int jmax = 0;
    private final int ACCEL_MAX = 1;
    private final int ACCEL_MIN = -1;
    private final double ALPHA = 0.001;
    private final double EPSILON = 0.05;
    private final double GAMMA = 0.95;
    private final int SIMS = 10000;
    private final int MAXMOVES = 1000;
    
    private final String filename = "tracks/left.track.improved";
    private final boolean VISUALIZE = true;
    private final boolean VERBOSE = true;

    private double[][][][] valueArray;
    private Random r;

    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_BLACK = "\u001B[30m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_YELLOW = "\u001B[33m";
    public static final String ANSI_BLUE = "\u001B[34m";
    public static final String ANSI_PURPLE = "\u001B[35m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_WHITE = "\u001B[37m";

    private static final String BORDER = ANSI_YELLOW + "**********" + ANSI_RESET;

    private RaceTrackSim episode;

    private double[][] Q;

    public QLearning() {
        // To get the neural network working
        /*Fann fann = new Fann( "/path/to/file" );
        float[] inputs = new float[]{ -1, 1 };
        float[] outputs = fann.run( inputs );
        fann.close();*/
        if (VISUALIZE) {
            readTrack();
        }

        this.episode = new RaceTrackSim();
        this.r = new Random();
        Q = new double[9][5];

        prf("\n\t\t%s Starting simulations to learn the optimal policy with exploring. %s\n\n", BORDER, BORDER);
        sleep(1000);
        long startTime = System.currentTimeMillis();
        for (int sims = 0; sims < SIMS; sims++) {
            simulate(true);
        }
        long endTime = System.currentTimeMillis();
        prf("\n\t\t%s Time spent learning policy %.2fs. %s\n\n", BORDER, (endTime-startTime)/1000.0, BORDER);


        prf("\n\t\t%s Printing the learned Q policy array. %s\n\n", BORDER, BORDER);
        for (double[] accelerations : Q) {
            for (double constant : accelerations) {
                prf("%.2f ", constant);
            }
            prf("\n");
        }

        prf("\n\t\t%s Starting simulations using the optimal policy with no exploring. %s\n\n", BORDER, BORDER);
        sleep(1000);
        double totalScore = 0;
        for (int optimals = 0; optimals < SIMS; optimals++) {
            totalScore += simulate(false);
        }
        prf("\n\t\t%s Average score of the optimal policy runs is %.2f. %s\n\n", BORDER, totalScore / SIMS, BORDER);
    }

    private class Move {
        public int ai;
        public int aj;
        public double value;
        Move(int ai, int aj, double value) {
            this.ai = ai;
            this.aj = aj;
            this.value = value;
        }
    }

    int round(double a) {
        return (int)Math.round(a);
    }

    int normaind(int ai, int aj) {
        int normai = ai + 1;
        int normaj = aj + 1;
        return normaj + normai * 3;
    }

    double dot(double[] a, double[] b) {
        double total = 0;
        for (int i = 0; i < a.length; i++) {
            total += a[i] * b[i];
        }
        return total;
    }

    double[] plus(double[] a, double[] b) {
        double[] result = a.clone();
        for (int i = 0; i < b.length; i++) {
            result[i] += b[i];
        }
        return result;
    }

    double[] mult(double a, double[] b) {
        double[] result = b.clone();
        for (int i = 0; i < b.length; i++) {
            result[i] *= a;
        }
        return result;
    }

    void prf(String format, Object... arguments) {
        System.out.printf(format, arguments);
    }

    //Prints the track in pretty colors
    void printTrack(int carj, int cari) {
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
        sleep(100);
    }

    void readTrack () {
        try {
            Scanner in = new Scanner(new File(filename));
            this.jmax = in.nextInt();
            this.imax = in.nextInt();
            System.out.println(jmax + " " + imax);
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
    }

    void sleep(int milliseconds) {
        try {
            TimeUnit.MILLISECONDS.sleep(milliseconds);
        } catch(InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }

    private double Q(double i, double j, double vi, double vj, int ai, int aj) {
        // System.out.printf("in Q with: %f %f %f %f\n", i, j, vi, vj);
        int aind = normaind(ai, aj);
        return dot(this.Q[aind], new double[] { 1, i, j, vi, vj });
    }

    private void updateQ(double i, double j, double vi, double vj, int ai, int aj, int reward, int terminal) {
        int aind = normaind(ai, aj);
        double correction = ALPHA * (reward + GAMMA * bestMove(i, j, vi + ai, vj + aj).value - Q(i, j, vi, vj, ai, aj));
        if (terminal == 1) {
            correction = ALPHA * (reward - Q(i, j, vi, vj, ai, aj));
        }
        // System.out.printf("in Q with: %f %f %f %f\n", i, j, vi, vj);
	//prf("%f\n", correction);
        this.Q[aind] = plus(this.Q[aind], mult(correction, new double[] { 1, i, j, vi, vj }));
    }

    private Move bestMove(double i, double j, double vi, double vj) {
        double best = Double.NEGATIVE_INFINITY;
        int bestai = 0, bestaj = 0;
        for (int ai = ACCEL_MIN; ai <= ACCEL_MAX; ai++) {
            for (int aj = ACCEL_MIN; aj <= ACCEL_MAX; aj++) {
                double actionValue = Q(i, j, vi, vj, ai, aj);
                // System.out.printf("Action value: %f\n", actionValue);
                if (actionValue > best) {
                    best = actionValue;
                    bestai = ai;
                    bestaj = aj;
                }
            }
        }
        return new Move(bestai, bestaj, best);
    }

    private Move randomMove() {
        return new Move(r.nextInt(3) - 1, r.nextInt(3) - 1, 0);
    }

    private double simulate(boolean explore) {
        double [] currState = episode.startEpisode();
        int score = 0;
        int terminated = 0;
        int b = 0;
        while (terminated != 1 && b < MAXMOVES) {
            b++;
            double i =  currState[0], j =  currState[1], vi =  currState[2], vj =  currState[3];
            Move best = bestMove(i, j, vi, vj);
            double chance = r.nextDouble();
            if (explore && chance < EPSILON) {
                // prf("Take random move \n");
                best = randomMove();
            }
            // System.out.printf("From (%f, %f) with speed (%f, %f) we take acceleration (%d, %d)\n", i, j, vi, vj, best.ai, best.aj);
            double[] result = episode.simulate(i, j, vi, vj, best.ai, best.aj);
            int reward = round(result[4]);
            terminated = round(result[5]);
            // System.out.printf("Reward we get is %d\n", reward);


            score += reward;
            // Update the current positions
            currState = result;
            // Learn
            if(explore)
                updateQ(i, j, vi, vj, best.ai, best.aj, reward, terminated);
            if (VISUALIZE && !explore) {
                printTrack(round(i), round(j));
            }
        }
        if (b != MAXMOVES) {
            if(VERBOSE)
                prf("\t\t%sTerminated%s in %d moves with a score of %d!\n", ANSI_GREEN, ANSI_RESET, b, score);
        } else {
            if(VERBOSE)
                prf("\t\t%sFailed%s to terminate in %d moves with a score of %d\n", ANSI_RED, ANSI_RESET, b, score);
        }
        return score;
    }

    private double max(double a, double b) {
        return a < b ? b : a;
    }

    private double abs(double a) {
        return a < 0 ? -a : a;
    }

    public static void main(String[] args) throws Exception {
        new QLearning();
    }
}
