package minicpbp.examples;

import minicpbp.engine.core.Constraint;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.util.exception.InconsistencyException;
import static minicpbp.cp.Factory.*;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.json.JSONObject;
import org.json.JSONException;

import minicpbp.examples.TrafficCrossingDecomposition.CrossingTrafficModel;

/**
 * TCP server exposing a costRegular-based CP model for CrossingTraffic (ETR mode).
 *
 * Wire protocol:
 *   INIT <instance_id>                          -> OK INIT successful for <id>
 *   RESET                                       -> OK RESET successful
 *   STEP <k> <m> <robot_idx> <obs_mask>         -> OK STEP processed
 *       k          : step number, must equal currentEpisodeStep
 *       m          : move action ∈ {0..4} (NORTH, SOUTH, EAST, WEST, NOOP)
 *       robot_idx  : observed robot position AFTER applying m (= R_{k+1});
 *                    -1 means DEAD
 *       obs_mask   : bitmask of W*H bits, bit i = obstacle at cell i at step k
 *                    (the observation snapshot used by the decision m)
 *   QUERY_ETR                                   -> ETR_VALUE <float>
 *   QUIT                                        -> OK Goodbye
 *
 * Model overview (T = nbSteps):
 *   R[0..T]          robot position, R[0] = init (asserted at RESET)
 *   O[0..T-1][i]     obstacle indicator at cell i at step k ∈ {0,1}
 *   move[0..T-1]     ∈ {0..4}
 *   collision[0..T-1] ∈ {0,1}, derived from ShortTableCT(R[k], O[k], collision[k])
 *   action[0..T-1]   = move[k] + 5*collision[k]   (Table)
 *   totalCost        sum of -1 per non-goal state along R[0..T-1] (costRegular).
 *                    DEAD is absorbing but still pays -1/step (matches IPPC).
 *
 * Exogenous stochasticity: Oracle marginals on input-border cells (x = W-1)
 * of INNER rows only (y ∈ [1, H-2]). Safe rows (y = 0 and y = H-1) are
 * assigned 0 directly — we never post an oracle there to avoid an apparent
 * conflict (oracle marginal {1-rate, rate} immediately squashed by assign(0)).
 *
 * Default port: 12346.  Config file: instances.json.
 */
public class TrafficCrossingService {

    private static int PORT = 12346;
    private static final String INSTANCES_JSON_FILE = "crossingtraffic_mdp_ippc2011.json";
    private static int BP_ITERATIONS = 1;
    private static boolean STEP_DEBUG = false;   // -Dstep.debug=true to enable

    // -------------------------------------------------------------------------
    // Instance parameters (loaded on INIT)
    // -------------------------------------------------------------------------

    private static int    W         = -1;
    private static int    H         = -1;
    private static double inputRate = Double.NaN;
    private static int    nbSteps   = -1;
    private static int    goalX     = -1;
    private static int    goalY     = -1;
    private static int    initX     = -1;
    private static int    initY     = -1;
    private static int    cells     = -1;   // W * H

    // -------------------------------------------------------------------------
    // CP model (rebuilt on each RESET)
    // -------------------------------------------------------------------------

    private static Solver    cp;
    private static IntVar[]   R;         // [0..nbSteps]
    private static IntVar[][] O;         // [0..nbSteps-1][cells]
    private static IntVar[]   move;      // [0..nbSteps-1]
    private static IntVar[]   collision; // [0..nbSteps-1]
    private static IntVar[]   action;    // [0..nbSteps-1]
    private static IntVar     totalCost;
    private static int        currentEpisodeStep = 0;
    private static CrossingTrafficModel model;

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
        STEP_DEBUG = Boolean.parseBoolean(System.getProperty("step.debug", "false"));

        System.out.println("TrafficCrossing CP Server (costRegular) starting on port " + PORT
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
            System.out.println("TrafficCrossing CP Server listening on port " + PORT);

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
                                    } else if (tokens.length == 5) {
                                        response = handleStep(tokens[1], tokens[2], tokens[3], tokens[4]);
                                    } else {
                                        response = "ERROR Invalid STEP format. Expected: STEP <k> <m> <robot_idx> <obs_mask>";
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
            W         = d.getInt("grid_x");
            H         = d.getInt("grid_y");
            inputRate = d.getDouble("input_rate");
            nbSteps   = d.getInt("cp_nbSteps");
            cells     = W * H;

            // Standard IPPC CrossingTraffic layout: init bottom-left, goal top-right.
            // (y grows upward; safe rows are y=0 and y=H-1.)
            // Optional JSON overrides: goal_x, goal_y, init_x, init_y.
            goalX = d.has("goal_x") ? d.getInt("goal_x") : (W - 1);
            goalY = d.has("goal_y") ? d.getInt("goal_y") : (H - 1);
            initX = d.has("init_x") ? d.getInt("init_x") : (W -1);
            initY = d.has("init_y") ? d.getInt("init_y") : 0;

            System.out.println("Loaded instance '" + instanceId + "': W=" + W + " H=" + H
                + " rate=" + inputRate + " nbSteps=" + nbSteps
                + " init=(" + initX + "," + initY + ") goal=(" + goalX + "," + goalY + ")");
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
        System.out.println("Handling RESET (CrossingTraffic costRegular, " + W + "x" + H + ")...");
        try {
            cp = makeSolver();
            model = TrafficCrossingDecomposition.build(W, H, goalX, goalY, initX, initY);

            int DEAD = model.deadState;

            // ---- Variables ----
            // R[k] ∈ [0, cells] : positions ∪ {DEAD = cells}.
            //R         = new IntVar[nbSteps + 1];
            //for (int k = 0; k <= nbSteps; k++) {
            //    R[k] = makeIntVar(cp, 0, cells);  // domain {0..cells}
            //}
            R         = makeIntVarArray(cp, nbSteps+1, cells+1); // cells+1 car il faut tenir compte du state dead 
            move      = makeIntVarArray(cp, nbSteps, TrafficCrossingDecomposition.NB_MOVES);
            collision = makeIntVarArray(cp, nbSteps, 2);
            action    = makeIntVarArray(cp, nbSteps, TrafficCrossingDecomposition.NB_ACTIONS);
            totalCost = makeIntVar(cp, -nbSteps, 0);
            O         = new IntVar[nbSteps][cells];

            for (int k = 0; k < nbSteps; k++) {
                O[k] = makeIntVarArray(cp, cells, 2);
            }

            // ---- Initial state ----
            R[0].assign(model.initState);

            // ---- Obstacle dynamics ----
            int[][] actionEncoding = TrafficCrossingDecomposition.buildActionEncodingTable();
            int[][] collisionTuples = TrafficCrossingDecomposition.buildCollisionTableTuples(W, H);

            for (int k = 0; k < nbSteps; k++) { // Pour tous les temps
                // Safe rows: y = 0 and y = H-1 never carry cars.
                for (int x = 0; x < W; x++) { // Pour tous les x
                    O[k][TrafficCrossingDecomposition.idx(x,0, W)].assign(0);
                    O[k][TrafficCrossingDecomposition.idx(x, H - 1, W)].assign(0);
                }
                // Input border (x = W-1) on INNER rows: oracle with (1-rate, rate).
                // No oracle on safe rows — they're already pinned to 0 above.
                int[] vals      = new int[]{0, 1};
                double[] margs  = new double[]{1.0 - inputRate, inputRate};
                for (int y = 1; y <= H - 2; y++) {  // Toutes les colonnes qui ont des voitures
                    cp.post(oracle(O[k][TrafficCrossingDecomposition.idx(W - 1, y, W)], vals, margs));
                }
                // Left-shift dynamics from step k to k+1: inner columns x < W-1.
                if (k + 1 < nbSteps) {
                    for (int y = 1; y <= H - 2; y++) { // Toutes les colonnes concernées
                        for (int x = 0; x < W - 1; x++) { // On arrête un x avant car le dernier c'est l'oracle
                            cp.post(equal(
                                O[k + 1][TrafficCrossingDecomposition.idx(x, y, W)],
                                O[k][TrafficCrossingDecomposition.idx(x + 1, y, W)]));
                        }
                    }
                }

                // ---- Collision link: scope [R[k], O[k][0..cells-1], collision[k]] ----
                IntVar[] scope = new IntVar[cells + 2];
                scope[0] = R[k];
                for (int i = 0; i < cells; i++) scope[1 + i] = O[k][i];
                scope[cells + 1] = collision[k];
                cp.post(shortTable(scope, collisionTuples, TrafficCrossingDecomposition.STAR));

                // ---- Action encoding: [move, collision, action] ----
                IntVar[] triple = new IntVar[]{move[k], collision[k], action[k]};
                cp.post(table(triple, actionEncoding));
            }

            // ---- Global costRegular over the action sequence ----
            // All states accepting; initial state = init position. Costs 2D.
            List<Integer> allAccepting = new ArrayList<>();
            for (int s = 0; s < model.nbStates; s++) allAccepting.add(s);
            
            cp.post(costRegular(action, model.A, model.initState, allAccepting,
                                model.cost, totalCost, /*marginals4cost=*/true));

            currentEpisodeStep = 0;
            cp.fixPoint();
            System.out.println("CrossingTraffic CP model reset successfully.");
            return "OK RESET successful";

        } catch (Exception e) {
            System.err.println("Error during RESET: " + e.getMessage());
            e.printStackTrace();
            cp = null;
            return "ERROR RESET failed: " + e.getMessage();
        }
    }

    static String handleStep(String kStr, String mStr, String rStr, String obsStr) {
        if (cp == null) return "ERROR Must RESET first";
        try {
            int k         = Integer.parseInt(kStr);
            int m         = Integer.parseInt(mStr);
            int rIdx      = Integer.parseInt(rStr);
            long obsMask  = Long.parseLong(obsStr);

            if (k != currentEpisodeStep)
                return "ERROR STEP index mismatch. Expected " + currentEpisodeStep + ", got " + k;
            if (k < 0 || k >= nbSteps)
                return "ERROR Step index out of bounds [0.." + (nbSteps - 1) + "]";
            if (m < 0 || m >= TrafficCrossingDecomposition.NB_MOVES)
                return "ERROR Move out of bounds [0.." + (TrafficCrossingDecomposition.NB_MOVES - 1) + "]";
            if (rIdx < -1 || rIdx >= cells)
                return "ERROR robot_idx out of bounds [-1.." + (cells - 1) + "]";

            // Fix the move
            move[k].assign(m);

            // Fix observed obstacles at step k
            for (int i = 0; i < cells; i++) {
                int b = (int) ((obsMask >> i) & 1L);
                O[k][i].assign(b);
            }

            // Fix observed next position R[k+1]
            int nextR = (rIdx == -1) ? model.deadState : rIdx;
            R[k + 1].assign(nextR);

            cp.fixPoint();

            if (STEP_DEBUG) {
                StringBuilder obsList = new StringBuilder();
                for (int i = 0; i < cells; i++) {
                    if (O[k][i].isBound() && O[k][i].min() == 1) {
                        if (obsList.length() > 0) obsList.append(",");
                        obsList.append(i);
                    }
                }
                int rNow = R[k].isBound() ? R[k].min() : -2;
                System.err.println("[STEP_DEBUG] k=" + k + " move=" + m
                    + " R[k]=" + rNow + " R[k+1]=" + nextR
                    + " O[k]={" + obsList + "}");
            }

            currentEpisodeStep++;
            return "OK STEP processed";

        } catch (InconsistencyException e) {
            System.err.println("Inconsistency on STEP " + kStr + ": " + e.getMessage());
            return "ERROR Inconsistency STEP " + kStr;
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

            double etr = 0.0;
            for (int v = totalCost.min(); v <= totalCost.max(); v++) {
                if (totalCost.contains(v)) {
                    etr += v * totalCost.marginal(v);
                }
            }
            return "ETR_VALUE " + etr;

        } catch (InconsistencyException e) {
            System.err.println("WARN: Inconsistency during QUERY_ETR. Returning 0.");
            return "ETR_VALUE 0.0";
        } catch (Exception e) {
            System.err.println("ERROR getting ETR: " + e.getMessage());
            e.printStackTrace();
            return "ETR_VALUE 0.0";
        }
    }
}
