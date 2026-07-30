/*
 * mini-cp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * mini-cp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2018. by Laurent Michel, Pierre Schaus, Pascal Van Hentenryck
 */

package minicpbp.examples.MiniCPBP_Example;

import minicpbp.cp.Factory;
import minicpbp.engine.core.IntVar;
import minicpbp.engine.core.Solver;
import minicpbp.search.DFSearch;
import minicpbp.search.SearchStatistics;
import minicpbp.util.io.InputReader;

import java.util.Arrays;

import static minicpbp.cp.BranchingScheme.and;
import static minicpbp.cp.BranchingScheme.firstFail;
import static minicpbp.cp.Factory.*;

/**
 *
 *  The Eternity II puzzle is an edge-matching puzzle which
 *  involves placing 256 square puzzle pieces into a 16 by 16 grid,
 *  constrained by the requirement to match adjacent edges.
 *  <a href="https://en.wikipedia.org/wiki/Eternity_II_puzzle">Wikipedia.</a>
 */
public class Eternity {

    public static IntVar[] flatten(IntVar[][] x) {
        return Arrays.stream(x).flatMap(Arrays::stream).toArray(IntVar[]::new);
    }

    public static void main(String[] args) {

        // Reading the data

        InputReader reader = new InputReader("data/eternity/eternity7x7.txt");

        int n = reader.getInt();
        int m = reader.getInt();

        int[][] pieces = new int[n * m][4];
        int maxTmp = 0;

        for (int i = 0; i < n * m; i++) {
            for (int j = 0; j < 4; j++) {
                pieces[i][j] = reader.getInt();
                if (pieces[i][j] > maxTmp)
                    maxTmp = pieces[i][j];
            }
            System.out.println(Arrays.toString(pieces[i]));
        }
        final int max = maxTmp;

        // ------------------------

        // TODO: create the table where each line correspond to one possible rotation of a piece
        // For instance if the line piece[6] = [2,3,5,1]
        // the four lines created in the table are
        // [6,2,3,5,1] // rotation of 0°
        // [6,3,5,1,2] // rotation of 90°
        // [6,5,1,2,3] // rotation of 180°
        // [6,1,2,3,5] // rotation of 270°

        // Table with makeIntVarArray pieces and for each their 4 possible rotations

        int[][] table = new int[4 * n * m][5];
        for (int i = 0; i < n * m; i++) {
            for (int rot = 0; rot < 4; rot++) {
                table[4 * i + rot][0] = i; // id de la pièce
                for (int side = 0; side < 4; side++) {
                    table[4 * i + rot][1 + side] = pieces[i][(side + rot) % 4];
                }
            }
        }

        Solver cp = makeSolver();

        //   |         |
        // - +---------+- -
        //   |    u    |
        //   | l  i  r |
        //   |    d    |
        // - +---------+- -
        //   |         |


        IntVar[][] id = new IntVar[n][m]; // id
        IntVar[][] u = new IntVar[n][m];  // up
        IntVar[][] r = new IntVar[n][m];  // right
        IntVar[][] d = new IntVar[n][m];  // down
        IntVar[][] l = new IntVar[n][m];  // left

        for (int i = 0; i < n; i++) {
            u[i] = Factory.makeIntVarArray(m, j -> makeIntVar(cp, 0, max));
            id[i] = makeIntVarArray(cp, m, n * m);
        }
        for (int k = 0; k < n; k++) {
            final int i = k;
            if (i < n - 1) { //Le bord du bad de i c'est le bord du haut de i+1 
                d[i] = u[i + 1];
            }
            else {
                d[i] = Factory.makeIntVarArray(m, j -> makeIntVar(cp, 0, max));
            }
        }
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                l[i][j] = makeIntVar(cp, 0, max);
            }
        }
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                if (j < m - 1) r[i][j] = l[i][j + 1];
                else r[i][j] = makeIntVar(cp, 0, max);
            }
        }

        // The constraints of the problem

        // TODO: State the constraints of the problem
        // Constraint1: all the pieces placed are different
        cp.post(Factory.allDifferent(flatten(id)));
        // Constraint2: all the pieces placed are valid ones i.e. one of the given mxn pieces possibly rotated
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                cp.post(Factory.table(
                    new IntVar[]{id[i][j], u[i][j], r[i][j], d[i][j], l[i][j]},
                    table
                ));
            }
}
        // Constraint3: place "0" one all external side of the border (gray color)
        IntVar zero = Factory.makeIntVar(cp, 0, 0);
        for (int j = 0; j < m; j++) {
            cp.post(Factory.equal(u[0][j],zero));  // ligne du haut
            cp.post(Factory.equal(d[n-1][j],zero));  // ligne du bas
        }
        for (int i = 0; i < n; i++) {
            cp.post(Factory.equal(l[i][0],zero));  // colonne gauche
            cp.post(Factory.equal(r[i][m-1],zero));  // colonne droite
}
        


        // The search using the and combinator

        DFSearch dfs = makeDfs(cp,
                /* TODO: continue, are you branching on all the variables ? */
                and(
                    firstFail(flatten(id)), 
                    firstFail(flatten(u)),
                    firstFail(flatten(l)))
        );


        dfs.onSolution(() -> {
        for (int i = 0; i < n; i++) {
            // Ligne du haut (couleur u)
            String line = "   ";
            for (int j = 0; j < m; j++) {
                line += u[i][j].min() + "      ";
            }
            System.out.println(line);

            // Ligne du milieu (l, id, r)
            line = " ";
            for (int j = 0; j < m; j++) {
                line += l[i][j].min() + " [" + id[i][j].min() + "] ";
            }
            line += r[i][m - 1].min();
            System.out.println(line);

            // Ligne du bas (couleur d)
            line = "   ";
            for (int j = 0; j < m; j++) {
                line += d[i][j].min() + "      ";
            }
            System.out.println(line);

            System.out.println(); // séparation entre les lignes
        }
    });


        SearchStatistics stats = dfs.solve(statistics -> statistics.numberOfSolutions() == 1);

        System.out.format("#Solutions: %s\n", stats.numberOfSolutions());
        System.out.format("Statistics: %s\n", stats);

    }
}
