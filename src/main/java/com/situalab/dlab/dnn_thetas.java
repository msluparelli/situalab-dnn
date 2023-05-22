package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static java.lang.Math.sqrt;
import static java.lang.StrictMath.pow;


public class dnn_thetas {

    //attributes, architecture
    private int featsX;
    private int outlab;
    private int thetaslen;
    private int[] thetasArch;
    private int neurons;
    private int seedth;
    private double[] initThetas;
    private String initializer;
    private String fan_mod;

    //constructor
    public dnn_thetas(int featuresX_,
                      int[] neuronLayers_,
                      int outputY_,
                      int seedNN_,
                      String initializer_,
                      String fan_mod_){

        //alternative architecture

        //features
        this.featsX = featuresX_;
        int[] feats = {featsX};

        //outputs
        this.outlab = outputY_;
        int[] outpv = {outlab};

        //deeplab_thetas
        int[] thetasArch = IntStream.concat(Arrays.stream(feats), IntStream.concat(Arrays.stream(neuronLayers_), Arrays.stream(outpv))).toArray();
        this.thetaslen = 0; //deeplab_thetas count
        for (int j=1; j<thetasArch.length; j++){thetaslen += (thetasArch[j-1]+1)*thetasArch[j];}
        this.thetasArch = thetasArch;



        //neurons, params
        this.neurons = IntStream.of(neuronLayers_).sum()+outputY_; //neurons
        this.seedth = seedNN_;
        this.initializer = initializer_;
        this.fan_mod = fan_mod_;

        //System.out.println(initializer);

        //print architecture
        System.out.println(featsX + " features " + Arrays.toString(neuronLayers_) + " hidden layers " + outlab + " output unit");
        System.out.println(thetaslen + " weights, " + neurons + " neuronas (seed=" + seedth + " initmode "+initializer+" fan_mod "+fan_mod);

        //random deeplab_thetas, https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers/initializers
        Random rand = new Random();
        rand.setSeed(seedth);
        rand.longs(thetaslen);

        //uniform distribution
        double[] thetasXavierUniform = new double[thetaslen];
        double epsilon_initA = sqrt(6.0 / (featsX+outlab));

        //normal distribution
        double[] thetasXavierNormal = new double[thetaslen];
        double epsilon_initB = sqrt(3.0 / (featsX+outlab));

        //deeplearning
        double factor = 1.0; //2.0 o 1.0
        double n = 0.0;
        if (fan_mod.equals("fan_in")) n = featsX;
        if (fan_mod.equals("fan_out")) n = outlab;
        if (fan_mod.equals("fan_avg")) n = (featsX+outlab)/2.0;
        double stdev = sqrt(factor/n);
        double[] thetasDeepLearning = new double[thetaslen];

        //initalizer
        for (int i = 0; i < thetaslen; i++) {

            double wnn = rand.nextDouble();
            thetasXavierUniform[i] = wnn * 2 * epsilon_initA - epsilon_initA; //formula to initialize deeplab_thetas

            double wga = rand.nextGaussian()*epsilon_initB; //standard deviation
            thetasXavierNormal[i] = wga;

            double wfm = rand.nextGaussian()*stdev;
            thetasDeepLearning[i] = wfm;

        }

        if (initializer.equals("xuniform")) this.initThetas = thetasXavierUniform; //Xavier Uniform
        if (initializer.equals("xnormal")) this.initThetas = thetasXavierNormal; //Xavier Normal
        if (initializer.equals("deeplearning")) this.initThetas = thetasDeepLearning; //deeplearning Normal
    }



    //List of deeplab_thetas
    public List<double[][]> getThetasL(){

        int thstart = 0; //start
        List<double[][]> thetasList = new ArrayList<double[][]>();

        for (int layer = 1; layer <thetasArch.length; layer++){
            int row = thetasArch[layer-1]+1; //rows
            int col = thetasArch[layer]; //columns
            int thstop = thstart + (row*col); //stop
            double[] thetasArray = Arrays.copyOfRange(initThetas, thstart, thstop); //deeplab_thetas
            thstart = thstop; //update start

            int ij = 0;
            double[][] thetaslayer = new double[row][col];
            for (int j=0; j<col; j++){ for (int i=0; i<row; i++){
                thetaslayer[i][j] = thetasArray[ij];
                if (i==0) {thetaslayer[0][j] = thetasArray[ij]*0;} //bias init at zero (ReLU, 0.1)
                ij +=1;
            }
            }
            thetasList.add(thetaslayer);
        }
        return thetasList;
    }



    //Accumulate derivative deeplab_thetas
    public List<dnn_thetasAccumM> getAccThetas(List<double[][]> Thetas){
        List<dnn_thetasAccumM> WAccReturn = new ArrayList<dnn_thetasAccumM>();
        for (int layer=0; layer<Thetas.size(); layer++){
            int i = Thetas.get(layer).length;
            int j = Thetas.get(layer)[0].length;
            dnn_thetasAccumM WAcc = new dnn_thetasAccumM(i,j);
            WAccReturn.add(WAcc);
        }
        return WAccReturn;
    }





    //zero deeplab_thetas
    public List<double[][]> getThetasLZeros(List<double[][]> MList){
        List <double[][]> WeightsZeroList = new ArrayList<double[][]>();
        for (int layer=0; layer<MList.size(); layer++){
            int row = MList.get(layer).length;
            int col = MList.get(layer)[0].length;
            double[][] WZeros = new double[row][col];
            WeightsZeroList.add(WZeros);
        }
        return WeightsZeroList;
    }





    //transpose List of Thetas
    public List<double[][]> getThetasLTranspose(List<double[][]> weightsList){
        List<double[][]> weightsListTransposed = new ArrayList<double[][]>();
        for (int layer=0; layer<weightsList.size(); layer++){
            double [][] wT = getTranspose(weightsList.get(layer));
            weightsListTransposed.add(wT);
        }
        return weightsListTransposed;
    }





    //transpose Matrix
    public double[][] getTranspose(double[][] initMatrix){
        int row = initMatrix.length;
        int col = initMatrix[0].length;
        double[][] outputMatrix = new double[col][row];
        for (int j=0; j<col; j++) {
            for (int i = 0; i<row; i++) {
                outputMatrix[j][i] = initMatrix[i][j];
            }
        }
        return outputMatrix;
    }




    //get square roots of weights
    public double getthetassqw(List<double[][]> thetas){
        double sqw = 0;
        for (int layer=0; layer<thetas.size(); layer++){
            double[][] W = thetas.get(layer);
            //not including bias
            for(int i=1; i<W.length; i++){
                for (int j=0; j<W[0].length; j++){
                    sqw += pow(W[i][j],2);
                }
            }
        }
        return sqw;
    }





    //flatten Matrix
    public double[] getflatThetas(List<double[][]> thetasL){
        double[] thetasflat = new double[0];
        for (int layer=0; layer<thetasL.size(); layer++){
            double[] tflat = Arrays.stream(thetasL.get(layer)).flatMapToDouble(Arrays::stream).toArray();
            thetasflat = ArrayUtils.addAll(thetasflat, tflat);
        }
        return thetasflat;
    }

    //flatten Matrix
    public double[] getflatThetasAcc(List<dnn_thetasAccumM> dthetasL){
        double[] thetasflat = new double[0];
        for (int layer=0; layer<dthetasL.size(); layer++){
            double[] tflat = Arrays.stream(dthetasL.get(layer).value()).flatMapToDouble(Arrays::stream).toArray();
            thetasflat = ArrayUtils.addAll(thetasflat, tflat);
        }
        return thetasflat;
    }


    //get thetas trained
    public List<double[][]> gettrainedThetasL(double[] thetastrained){

        int thstart = 0; //start
        List<double[][]> thetasTrainedList = new ArrayList<double[][]>();

        for (int layer = 1; layer <thetasArch.length; layer++){
            int row = thetasArch[layer-1]+1; //rows
            int col = thetasArch[layer]; //columns
            int thstop = thstart + (row*col); //stop
            double[] thetasArray = Arrays.copyOfRange(thetastrained, thstart, thstop); //trained thetas
            thstart = thstop; //update start

            int ij = 0;
            double[][] thetaslayer = new double[row][col];
            for (int i=0; i<row; i++){
                for (int j=0; j<col; j++){
                    thetaslayer[i][j] = thetasArray[ij];
                    ij +=1;
                }
            }

            thetasTrainedList.add(thetaslayer);
        }
        return thetasTrainedList;
    }


    //get thetas+epsilon+gradient to compute Hd
    public List<double[][]> getthetasepsilon(List<double[][]> thetasL,
                                             List<double[][]> gthetas,
                                             List<double[][]> thetaseL){
        double epsilon = 1.e-4;
        for (int layer=0; layer<thetasL.size(); layer++){
            double[][] thetas = thetasL.get(layer);
            double[][] gradient = gthetas.get(layer);
            int row = thetas.length;
            int col = thetas[0].length;
            double[][] thetase = new double[row][col];

            for (int i=0; i<row; i++){
                for (int j=0; j<col ; j++){
                    thetase[i][j] = thetas[i][j]+(epsilon*-gradient[i][j]); //d = -gradient f(thetas)
                }
            }

            thetaseL.add(thetase);
        }


        return thetaseL;
    }


}
