package minicpbp.examples;

import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;
import static minicpbp.cp.Factory.*;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONException;

/**
 * TCP server exposing a per-cell CP model for the GameOfLife MDP (ETR mode).
 *
 * Wire protocol (same conventions as SysAdminService):
 *   INIT <instance_id>                              -> OK INIT successful for <id>
 *   RESET                                           -> OK RESET successful
 *   STEP <step_idx> <set_idx> <alive_bitmask>       -> OK STEP processed
 *       set_idx ∈ {0..N-1} : index of the cell to set
 *       set_idx = -1        : no set action this step
 *       alive_bitmask       : bit i = 1 iff cell i is alive after the step
 *   QUERY_ETR                                       -> ETR_VALUE <float>
 *   QUIT                                            -> OK Goodbye
 *
 * Model: one Markov constraint per cell, all sharing the same set action
 * variables sv[i][k]. The CP model is built fresh on each RESET.
 * Init state is instance-specific (loaded from config).
 *
 * Config file: gameoflife_instances.json (in the working directory).
 * Fields per instance:
 *   n_cells      : int
 *   noise_prob   : double[]  (RDDL NOISE-PROB per cell, in cell-index order)
 *   neighbors    : int[][]   (neighbors[i] = indices j with NEIGHBOR(i, j))
 *   init_state   : long      (bitmask: bit i = 1 iff cell i starts alive)
 *   horizon      : int
 *   cp_nbSteps   : int       (horizon + buffer)
 *
 * Default port: 12349.
 */
public class GameOfLifeService {

    private static int PORT = 12349;
    private static final String INSTANCES_JSON_FILE = "gameoflife_mdp_ippc2011.json";
    private static int BP_ITERATIONS = 1;

    // -------------------------------------------------------------------------
    // Instance parameters (loaded from JSON on INIT)
    // -------------------------------------------------------------------------

    private static int       N          = -1;
    private static int[][]   neighbors  = null;
    private static double[]  noiseProbs = null;
    private static long      initState  = 0L;
    private static int       horizon    = -1;
    private static int       nbSteps    = -1;

    // -------------------------------------------------------------------------
    // CP model (rebuilt on each RESET)
    // -------------------------------------------------------------------------

    private static Solver    cp                  = null;
    // sv[i][k] : set action for cell i at step k  ∈ {0,1}
    private static IntVar[][] sv                 = null;
    // n[i][k]  : alive neighbor count for cell i at step k  ∈ {0..deg(i)}
    private static IntVar[][] n                  = null;
    // a[i][k]  : encoded action = n*2 + sv  for cell i at step k
    private static IntVar[][] a                  = null;
    // s[i][k]  : alive state of cell i at step k  ∈ {0,1}
    private static IntVar[][] s                  = null;
    // per-cell total-reward variables (from Markov constraints)
    private static IntVar[] totalRewardPerCell   = null;
    private static int currentEpisodeStep        = 0;

    // -------------------------------------------------------------------------
    // Config
    // -------------------------------------------------------------------------

    private static JSONObject allInstancesConfig = null;
    private static String     currentInstanceId  = null;

    // -------------------------------------------------------------------------
    // Entry point
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        if (args.length > 0) {
            try { PORT = Integer.parseInt(args[0]); }
            catch (NumberFormatException e) {
                System.err.println("WARN: Invalid port '" + args[0] + "'. Using default " + PORT);
            }
        }
        String bpProp = System.getProperty("bp.iterations");
        if (bpProp != null) {
            try { BP_ITERATIONS = Integer.parseInt(bpProp); }
            catch (NumberFormatException e) {
                System.err.println("WARN: Invalid bp.iterations '" + bpProp + "'. Using default " + BP_ITERATIONS);
            }
        }

        System.out.println("GameOfLife CP Server starting on port " + PORT
            + " (bp.iterations=" + BP_ITERATIONS + ")");

        if (!loadAllInstancesConfig()) {
            System.err.println("FATAL: Could not load '" + INSTANCES_JSON_FILE + "'. Exiting.");
            System.exit(1);
        }
        System.out.println("Instance configurations loaded from " + INSTANCES_JSON_FILE);

        runServer();
    }

    // -------------------------------------------------------------------------
    // Server loop
    // -------------------------------------------------------------------------

    static void runServer() {
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("GameOfLife CP Server listening on port " + PORT);

            while (true) {
                try (Socket clientSocket = serverSocket.accept();
                     PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true);
                     BufferedReader in = new BufferedReader(
                             new InputStreamReader(clientSocket.getInputStream()))) {

                    System.out.println("\nClient connected: " + clientSocket.getInetAddress());
                    out.println("OK Welcome");

                    currentInstanceId = null;
                    cp = null;

                    String line;
                    while ((line = in.readLine()) != null) {
                        String[] tokens = line.trim().split("\\s+");
                        String command  = tokens.length > 0 ? tokens[0].toUpperCase() : "";
                        String response = "ERROR Unknown command '" + command + "'";

                        try {
                            switch (command) {
                                case "INIT":
                                    if (tokens.length == 2) {
                                        String id = tokens[1];
                                        if (loadInstanceParameters(id)) {
                                            currentInstanceId = id;
                                            response = "OK INIT successful for " + id;
                                            System.out.println("Initialized for instance: " + id);
                                        } else {
                                            response = "ERROR Failed loading instance " + id;
                                            currentInstanceId = null;
                                        }
                                    } else {
                                        response = "ERROR Invalid INIT format. Expected: INIT <instance_id>";
                                    }
                                    break;

                                case "RESET":
                                    response = (currentInstanceId == null)
                                        ? "ERROR Must INIT first"
                                        : handleReset();
                                    break;

                                case "STEP":
                                    if (currentInstanceId == null || cp == null) {
                                        response = "ERROR Must INIT and RESET first";
                                    } else if (tokens.length == 4) {
                                        response = handleStep(tokens[1], tokens[2], tokens[3]);
                                    } else {
                                        response = "ERROR Invalid STEP format. Expected: STEP <step_idx> <set_idx> <alive_bitmask>";
                                    }
                                    break;

                                case "QUERY_ETR":
                                    if (currentInstanceId == null || cp == null) {
                                        response = "ERROR Must INIT and RESET first";
                                    } else {
                                        response = handleQueryETR();
                                    }
                                    break;

                                case "QUIT":
                                    response = "OK Goodbye";
                                    System.out.println("Client requested QUIT.");
                                    break;

                                default:
                                    System.out.println("Unknown command: " + line);
                                    break;
                            }
                        } catch (Exception e) {
                            System.err.println("Error processing '" + line + "': " + e.getMessage());
                            e.printStackTrace();
                            response = "ERROR Processing failed: " + e.getClass().getSimpleName();
                        }

                        out.println(response);
                        if ("QUIT".equalsIgnoreCase(command)) break;
                    }

                } catch (IOException e) {
                    System.err.println("WARN: Client connection error: " + e.getMessage());
                } finally {
                    System.out.println("Client disconnected.");
                    currentInstanceId = null;
                    cp = null;
                }
            }
        } catch (IOException e) {
            System.err.println("FATAL: Server socket error on port " + PORT + ": " + e.getMessage());
            System.exit(1);
        }
    }

    // -------------------------------------------------------------------------
    // Config loading
    // -------------------------------------------------------------------------

    static boolean loadAllInstancesConfig() {
        try {
            String json = new String(Files.readAllBytes(Paths.get(INSTANCES_JSON_FILE)));
            allInstancesConfig = new JSONObject(json);
            return true;
        } catch (IOException e) {
            System.err.println("ERROR reading '" + INSTANCES_JSON_FILE + "': " + e.getMessage());
            System.err.println("  CWD: " + Paths.get(".").toAbsolutePath().normalize());
            return false;
        } catch (JSONException e) {
            System.err.println("ERROR parsing JSON: " + e.getMessage());
            return false;
        }
    }

    static boolean loadInstanceParameters(String instanceId) {
        if (allInstancesConfig == null) {
            System.err.println("ERROR: Config not loaded.");
            return false;
        }
        if (!allInstancesConfig.has(instanceId)) {
            System.err.println("ERROR: Instance '" + instanceId + "' not found.");
            return false;
        }
        try {
            JSONObject d = allInstancesConfig.getJSONObject(instanceId);
            N         = d.getInt("n_cells");
            initState = d.getLong("init_state");
            horizon   = d.getInt("horizon");
            nbSteps   = d.getInt("cp_nbSteps");

            JSONArray npJson = d.getJSONArray("noise_prob");
            noiseProbs = new double[N];
            for (int i = 0; i < N; i++)
                noiseProbs[i] = npJson.getDouble(i);

            JSONArray nbrsJson = d.getJSONArray("neighbors");
            neighbors = new int[N][];
            for (int i = 0; i < N; i++) {
                JSONArray row = nbrsJson.getJSONArray(i);
                neighbors[i]  = new int[row.length()];
                for (int j = 0; j < row.length(); j++)
                    neighbors[i][j] = row.getInt(j);
            }

            System.out.println("Loaded instance '" + instanceId + "': N=" + N
                + " horizon=" + horizon + " cp_nbSteps=" + nbSteps
                + " initState=" + Long.toBinaryString(initState));
            return true;
        } catch (JSONException e) {
            System.err.println("ERROR parsing instance '" + instanceId + "': " + e.getMessage());
            return false;
        }
    }

    // -------------------------------------------------------------------------
    // Command handlers
    // -------------------------------------------------------------------------

    static String handleReset() {
        System.out.println("Handling RESET (GameOfLife, N=" + N + ")...");
        try {
            cp = makeSolver();

            sv                = new IntVar[N][nbSteps];
            n                 = new IntVar[N][nbSteps];
            a                 = new IntVar[N][nbSteps];
            s                 = new IntVar[N][nbSteps];
            totalRewardPerCell = new IntVar[N];

            for (int i = 0; i < N; i++) {
                int deg = neighbors[i].length;
                sv[i] = makeIntVarArray(cp, nbSteps, 0, 1);
                n[i]  = makeIntVarArray(cp, nbSteps, 0, deg);
                a[i]  = makeIntVarArray(cp, nbSteps, 0, 2 * (deg + 1) - 1);
                s[i]  = makeIntVarArray(cp, nbSteps, 0, 1);
            }

            // Constraint 1 — at most one set action per step
            for (int k = 0; k < nbSteps; k++) {
                IntVar[] setsAtTimeK = new IntVar[N];
                for (int i = 0; i < N; i++) setsAtTimeK[i] = sv[i][k];
                IntVar setsAtTimeKSum = sum(setsAtTimeK);
                setsAtTimeKSum.removeAbove(1);
            }

            // Constraint 2 — neighbor count = sum of neighbors' states
            for (int i = 0; i < N; i++) {
                int[] nbrs = neighbors[i];
                if (nbrs.length == 0) {
                    for (int k = 0; k < nbSteps; k++)
                        n[i][k].assign(0);
                } else {
                    IntVar[] nbrStates = new IntVar[nbrs.length];
                    for (int k = 0; k < nbSteps; k++) {
                        for (int j = 0; j < nbrs.length; j++)
                            nbrStates[j] = s[nbrs[j]][k];
                        cp.post(sum(nbrStates, n[i][k]));
                    }
                }
            }

            // Constraint 3 — action encoding: (n[i][k], sv[i][k], a[i][k]) via Table
            for (int i = 0; i < N; i++) {
                int[][] tuples = GameOfLifeNodeDecomposition.buildActionEncodingTable(neighbors[i].length);
                for (int k = 0; k < nbSteps; k++) {
                    IntVar[] triple = {n[i][k], sv[i][k], a[i][k]};
                    cp.post(table(triple, tuples));
                }
            }

            // Constraint 4 — Markov transition per cell
            GameOfLifeNodeDecomposition.GameOfLifeMatrices m =
                GameOfLifeNodeDecomposition.buildAllMatrices(N, neighbors, noiseProbs);

            for (int i = 0; i < N; i++) {
                int cellInitState = (int)((initState >> i) & 1);
                IntVar trVar = makeIntVar(cp, -nbSteps, nbSteps);
                cp.post(markov(a[i], s[i], m.P_per_cell[i], m.R_per_cell[i], cellInitState, trVar));
                totalRewardPerCell[i] = trVar;
            }

            currentEpisodeStep = 0;
            cp.fixPoint();
            System.out.println("GameOfLife CP model reset successfully (" + N + " Markov constraints).");
            return "OK RESET successful";

        } catch (Exception e) {
            System.err.println("Error during RESET: " + e.getMessage());
            e.printStackTrace();
            cp = null;
            return "ERROR RESET failed: " + e.getMessage();
        }
    }

    static String handleStep(String stepStr, String setStr, String maskStr) {
        if (cp == null) return "ERROR Must RESET first";
        try {
            int k        = Integer.parseInt(stepStr);
            int sIdx     = Integer.parseInt(setStr);
            long aliveMask = Long.parseLong(maskStr);

            if (k != currentEpisodeStep)
                return "ERROR STEP index mismatch. Expected " + currentEpisodeStep + ", got " + k;
            if (k < 0 || k >= nbSteps)
                return "ERROR Step index out of bounds [0.." + (nbSteps - 1) + "]";
            if (sIdx < -1 || sIdx >= N)
                return "ERROR set_idx out of bounds [-1.." + (N - 1) + "]";

            // Fix set actions: exactly the chosen cell gets sv=1, others sv=0.
            for (int i = 0; i < N; i++) {
                sv[i][k].assign(i == sIdx ? 1 : 0);
            }

            // Fix observed next state
            if (k + 1 < nbSteps) {
                for (int i = 0; i < N; i++) {
                    int alive = (int)(aliveMask >> i) & 1;
                    s[i][k].assign(alive);
                }
            }

            currentEpisodeStep++;
            return "OK STEP processed";

        } catch (InconsistencyException e) {
            System.err.println("Inconsistency on STEP " + stepStr + ": " + e.getMessage());
            e.printStackTrace(System.err);
            return "ERROR Inconsistency STEP " + stepStr;
        } catch (NumberFormatException e) {
            return "ERROR Invalid number in STEP: " + e.getMessage();
        } catch (Exception e) {
            System.err.println("Unexpected error on STEP: " + e.getMessage());
            e.printStackTrace();
            return "ERROR Unexpected failure: " + e.getMessage();
        }
    }

    static String handleQueryETR() {
        if (cp == null) return "ERROR Must RESET first";
        try {
            cp.fixPoint();
            cp.vanillaBP(BP_ITERATIONS);

            double etrValue = 0.0;
            for (int i = 0; i < N; i++) {
                IntVar tr = totalRewardPerCell[i];
                for (int v = tr.min(); v <= tr.max(); v++) {
                    if (tr.contains(v)) {
                        double probability = tr.marginal(v);
                        etrValue += v * probability;
                    }
                }
            }

            System.err.println("ETR at step " + currentEpisodeStep + " = " + etrValue);
            return "ETR_VALUE " + etrValue;

        } catch (InconsistencyException e) {
            System.err.println("ERROR: Inconsistency during QUERY_ETR at step " + currentEpisodeStep);
            return "ERROR Inconsistency QUERY_ETR " + currentEpisodeStep;
        } catch (Exception e) {
            System.err.println("ERROR getting ETR: " + e.getMessage());
            e.printStackTrace();
            return "ETR_VALUE 0.0";
        }
    }
}
