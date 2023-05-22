package com.situalab.dlab;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;


public class dnn_thetaspretrained {

    public List<double[][]> thetasListpretrain(String thetasRpath) throws IOException {

        //read from local file
        Charset charset = Charset.forName("US-ASCII");
        Path thetaspath = Paths.get(thetasRpath);
        BufferedReader thetasreader = Files.newBufferedReader(thetaspath, charset); //cost file writter

        //read from resources
        //InputStream thetaspath = deeplab_thetaspretrain.class.getClassLoader().getResourceAsStream(thetasRpath);
        //BufferedReader thetasreader = new BufferedReader(new InputStreamReader(thetaspath));

        //thetas architecture
        String[] thetasParams = thetasreader.readLine().split("_");
        String[] thetasArchitecture = Arrays.copyOfRange(thetasParams, 0, (thetasParams.length-1));
        int[] thetasA = Stream.of(thetasArchitecture).mapToInt(Integer::parseInt).toArray(); //thetas Architecture
        //System.out.println(Arrays.toString(thetasA));
        String activation = thetasParams[thetasParams.length-1]; //activation function
        int thetaslen = 0; // thetas size
        for (int j=1; j<thetasA.length; j++){thetaslen += (thetasA[j-1]+1)*thetasA[j];} //thetas vector
        System.out.println("pretrained thetas loaded: "+activation+" "+Arrays.toString(thetasA)+" "+thetaslen); //print architecture


        //thetas values
        String thetavalue = null;
        double[] thetasvector = new double[thetaslen];
        double theta;
        int counter = 0;
        while ((thetavalue = thetasreader.readLine()) != null) {
            theta = Double.parseDouble(thetavalue);
            thetasvector[counter] = theta;
            counter +=1;
        }

        List<double[][]> thetasList = new ArrayList<double[][]>();

        int thstart = 0; //start


        for (int layer = 1; layer <thetasA.length; layer++){
            int row = thetasA[layer-1]+1; //rows
            int col = thetasA[layer]; //columns
            int thstop = thstart + (row*col); //stop
            double[] thetasArray = Arrays.copyOfRange(thetasvector, thstart, thstop); //trained thetas
            thstart = thstop; //update start

            int ij = 0;
            double[][] thetaslayer = new double[row][col];
            for (int i=0; i<row; i++){
                for (int j=0; j<col; j++){
                    thetaslayer[i][j] = thetasArray[ij];
                    ij +=1;
                }
            }

            thetasList.add(thetaslayer);
        }


        return thetasList;
    }


}
