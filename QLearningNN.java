import java.util.Scanner;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.Vector;
import java.util.Random;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.io.IOUtils;



import com.googlecode.fannj.*;


public class QLearningNN {
    private final int ACCEL_MAX = 1;
    private final int ACCEL_MIN = -1;
    private final float ALPHA = 0.0001f;
    private final float EPSILON = 0.25f;
    private final float GAMMA = 0.85f;
    private final int SIMS = 1000;
    private final int MAXMOVES = 25;

    private Random r;

    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_BLACK = "\u001B[30m";
    private static final String ANSI_RED = "\u001B[31m";
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_YELLOW = "\u001B[33m";
    private static final String ANSI_BLUE = "\u001B[34m";
    private static final String ANSI_PURPLE = "\u001B[35m";
    private static final String ANSI_CYAN = "\u001B[36m";
    private static final String ANSI_WHITE = "\u001B[37m";
    private static final String BORDER = ANSI_YELLOW + "**********" + ANSI_RESET;

    private RaceTrackSim episode;
    private Fann fann;

    private float[][] trainingInputsBuffer;
    private float[] trainingOutputBuffer;
    private int trainingInstances = 0;

    public QLearningNN() {
        // To get the neural network working
        /*ArrayList<Layer> layers = new ArrayList<Layer>();
        layers.add(Layer.create(6));
        layers.add(Layer.create(6, ActivationFunction.FANN_SIGMOID_SYMMETRIC));
        layers.add(Layer.create(1, ActivationFunction.FANN_SIGMOID_SYMMETRIC));
        this.fann = new Fann(layers);*/
        this.fann = new Fann("neural_network_optimal1.0.fann");

        // Initialize the buffers
        trainingInputsBuffer = new float[MAXMOVES * 10][6];
        trainingOutputBuffer = new float[MAXMOVES * 10];

        this.episode = new RaceTrackSim();
        this.r = new Random();

        prf("\n\t\t%s Starting simulations to learn the optimal policy with exploring. %s\n\n", BORDER, BORDER);
        sleep(1000);
        int terminatingRuns = 0;
        int failingRuns = 0;
        for (int sims = 0; sims < SIMS; sims++) {
            if (simulate(false) > 0)
                terminatingRuns++;
            else
                failingRuns++;
            if (failingRuns > 10) {
                failingRuns = 0;
                terminatingRuns = 0;
            }
            if (terminatingRuns >= 5) {
                // Comment the following line if you want to run the full simulations every time.
                // This ensures that if we have 10 failing runs and 5 terminating runs in a row we terminate
                // since the policy is likely to be good at this time point.
                break;
            }
        }

        prf("\n\t\t%s Starting simulations using the optimal policy with no exploring. %s\n\n", BORDER, BORDER);
        sleep(1000);
        float totalScore = 0;
        for (int optimals = 0; optimals < SIMS; optimals++) {
            totalScore += simulate(false);
        }
        fann.save("neural_network_optimal" + totalScore / SIMS + ".fann");
        fann.close();
        prf("\n\t\t%s Percentage of simulations which terminate is %.4f. %s\n\n", BORDER, totalScore / SIMS, BORDER);
    }

    private class Move {
        public int ai;
        public int aj;
        public float value;
        Move(int ai, int aj, float value) {
            this.ai = ai;
            this.aj = aj;
            this.value = value;
        }
    }

    int round(float a) {
        return (int)Math.round(a);
    }

    int normaind(int ai, int aj) {
        int normai = ai + 1;
        int normaj = aj + 1;
        return normaj + normai * 3;
    }

    float dot(float[] a, float[] b) {
        float total = 0;
        for (int i = 0; i < a.length; i++) {
            total += a[i] * b[i];
        }
        return total;
    }

    float[] plus(float[] a, float[] b) {
        float[] result = a.clone();
        for (int i = 0; i < b.length; i++) {
            result[i] += b[i];
        }
        return result;
    }

    float[] mult(float a, float[] b) {
        float[] result = b.clone();
        for (int i = 0; i < b.length; i++) {
            result[i] *= a;
        }
        return result;
    }

    void prf(String format, Object... arguments) {
        System.out.printf(format, arguments);
    }

    void sleep(int milliseconds) {
        try {
            TimeUnit.MILLISECONDS.sleep(milliseconds);
        } catch(InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }

    void batchTrain() {
        prf("Batch training in progress\n");
        Charset cs = Charset.forName("US-ASCII");
        BufferedWriter output = null;
        try {
            output = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("trainingdata.txt"), cs));
            output.write(Integer.toString(trainingInputsBuffer.length));
            output.write(" 6 1\n");
            for (int i = 0; i < trainingInputsBuffer.length; i++) {
                output.write(Float.toString(trainingInputsBuffer[i][0]));
                for (int j = 1; j < 6; j++) {
                    output.write(" " + Float.toString(trainingInputsBuffer[i][j]));
                }
                output.write("\n");
                output.write(Float.toString(trainingOutputBuffer[i]));
                output.write("\n");
            }
            output.close();
            Trainer train = new Trainer(this.fann);
            File temp = File.createTempFile("trainingdata", ".txt");
            temp.deleteOnExit();

            Trainer trainer = new Trainer(fann);
            trainer.setTrainingAlgorithm(TrainingAlgorithm.FANN_TRAIN_BATCH);
            float desiredError = 0.0000001f;
            IOUtils.copy(this.getClass().getResourceAsStream("trainingdata.txt"),  new FileOutputStream(temp));
            float mse = trainer.train(temp.getPath(), 3000, 1500, desiredError);
            prf("Current mse: %.10f\n", mse);
        } catch(IOException e) {
            prf("Error opening the file: %s.\n", e.getMessage());
        }
    }

    void addTrainingInstance(float[] inputs, float output) {
        if (trainingInstances == trainingOutputBuffer.length) {
            batchTrain();
            trainingInstances = 0;
        }
        trainingInputsBuffer[trainingInstances] = inputs.clone();
        trainingOutputBuffer[trainingInstances] = output;
        trainingInstances++;
    }

    private float Q(float i, float j, float vi, float vj, int ai, int aj) {
        float[] result = this.fann.run(new float[] { i, j, vi, vj, ai, aj });
        return result[0];
    }

    private void updateQ(float i, float j, float vi, float vj, int ai, int aj, int reward, int terminal) {
        float goal = reward + (GAMMA * bestMove(i, j, vi + ai, vj + aj * 20).value);
        if (terminal == 1) {
            goal = reward;
        }
        // prf("in Q with: %f %f %f %f\n", i, j, vi, vj);
        // prf("Goal: %f, current: %f\n", goal, Q(i, j, vi, vj, ai, aj));
        addTrainingInstance(new float[] { i, j, vi, vj, ai, aj }, goal / 20);
    }

    private Move bestMove(float i, float j, float vi, float vj) {
        float best = Float.NEGATIVE_INFINITY;
        int bestai = 0, bestaj = 0;
        for (int ai = ACCEL_MIN; ai <= ACCEL_MAX; ai++) {
            for (int aj = ACCEL_MIN; aj <= ACCEL_MAX; aj++) {
                float actionValue = Q(i, j, vi, vj, ai, aj);
                // prf("Action value from (%f, %f, %f, %f) with (%d, %d): %f\n", i, j, vi, vj, ai, aj, actionValue);
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

    private int simulate(boolean explore) {
        double [] currStateD = episode.startEpisode();
        float[] currState = new float[currStateD.length];
        for (int i = 0; i < currStateD.length; i++) {
            currState[i] = (float)currStateD[i];
        }
        int score = 0;
        int terminated = 0;
        int b = 0;
        while (terminated != 1 && ++b < MAXMOVES) {
            float i =  currState[0], j =  currState[1], vi =  currState[2], vj =  currState[3];
            Move best = bestMove(i, j, vi, vj);
            Double chance = r.nextDouble();
            if (explore && chance < EPSILON) {
                // prf("Take random move \n");
                best = randomMove();
            }
            // System.out.printf("From (%f, %f) with speed (%f, %f) we take acceleration (%d, %d) with value %f\n", i, j, vi, vj, best.ai, best.aj, best.value);
            double[] resultD = episode.simulate(i, j, vi, vj, best.ai, best.aj);
            float[] result = new float[resultD.length];
            for (int a = 0; a < resultD.length; a++) {
                result[a] = (float)resultD[a];
            }
            int reward = round(result[4]);
            terminated = round(result[5]);
            // System.out.printf("Reward we get is %d\n", reward);

            score += reward;
            // Update the current positions
            currState = result;
            // Learn
            if (explore) {
                updateQ(i, j, vi, vj, best.ai, best.aj, reward, terminated);
            }
        }
        if (b != MAXMOVES) {
            prf("\t\t%sTerminated%s in %d moves with a score of %d!\n", ANSI_GREEN, ANSI_RESET, b, score);
        } else {
            prf("\t\t%sFailed%s to terminate in %d moves with a score of %d\n", ANSI_RED, ANSI_RESET, b, score);
        }
        return terminated;
    }

    private float max(float a, float b) {
        return a < b ? b : a;
    }

    private float abs(float a) {
        return a < 0 ? -a : a;
    }

    public static void main(String[] args) throws Exception {
        new QLearningNN();
    }
}
