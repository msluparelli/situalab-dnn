package com.situalab.dlab;

import java.io.Serializable;


public class dnn_mops implements Serializable {

    //print double[][]
    public void printMatrix(double[][] matrixPrint){

        System.out.print("\nprint matrix method\n");
        for (int row = 0; row < matrixPrint.length; row++) {
            for (int column = 0; column < matrixPrint[row].length; column++) {
                System.out.print(matrixPrint[row][column] + " ");
            }
            System.out.println();
        }

    }

    //print Accumulator
    public void printAccMatrix(dnn_thetasAccumM matrixPrint){

        System.out.print("\nprint matrix method\n");
        for (int row = 0; row < matrixPrint.value().length; row++) {
            for (int column = 0; column < matrixPrint.value()[row].length; column++) {
                System.out.print(matrixPrint.value()[row][column] + " ");
            }
            System.out.println();
        }

    }


}
