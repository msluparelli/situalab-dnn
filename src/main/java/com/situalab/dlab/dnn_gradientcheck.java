package com.situalab.dlab;

import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.util.DoubleAccumulator;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;


public class dnn_gradientcheck {

    //attributes

    //constructor

    //methods

    public void checkGradient(LabeledPoint event){

        dnn_activation adF = new dnn_activation(); //activation functions
        dnn_feedback feedpropAlg = new dnn_feedback();

        int featuresN = event.features().size();
        double lmbda = 1.e-10;
        double epsilon = 10.e-4;
        int[] neuronsLayer = {4,4};
        int outputY = 1;
        String classmode = "bin";
        dnn_thetas thetasINIT = new dnn_thetas(featuresN, neuronsLayer, outputY, 754, "deeplearning", "fan_in");
        List<double[][]> thetasL = thetasINIT.getThetasL(); //thetas
        List<double[][]> thetasLt = thetasINIT.getThetasLTranspose(thetasL); //thetas transpose
        DoubleAccumulator costA = new DoubleAccumulator(); //cost
        DoubleAccumulator mproA = new DoubleAccumulator(); //events
        List<DoubleAccumulator> AccumList = new ArrayList<>();
        AccumList.add(costA); //add accumulator
        AccumList.add(mproA); //add accumulator
        List<dnn_thetasAccumM> dthetasAcc = new ArrayList<dnn_thetasAccumM>(); //accumulator
        dthetasAcc = thetasINIT.getAccThetas(thetasL); //local variables, no registrado en jsc()
        List<dnn_thetasAccumM> dthetasAccL = dthetasAcc;
        double yavg = 0;
        double mu = 0.95;
        List<double[][]> velocity = thetasINIT.getThetasLZeros(thetasL);
        String optimization = "momentum";


        dnn_activation.activateV<Double> actiF = adF.sigmoid;
        dnn_activation.activateV<Double> dactiF = adF.dsigmoid;

        feedpropAlg.getfeedbackprop("back", classmode, yavg, event, thetasLt, thetasL, actiF, dactiF, AccumList, dthetasAccL, mu, velocity, optimization);
        double[] dthetasV = thetasINIT.getflatThetasAcc(dthetasAccL);

        List<double[][]> gAprox = thetasINIT.getThetasLZeros(thetasL);

        for (int layer=0; layer<thetasL.size(); layer++){

            for (int i=0; i<thetasL.get(layer).length; i++){
                for (int j=0; j<thetasL.get(layer)[0].length; j++){

                    //thetaPlus
                    List<double[][]> thetaplus = thetasINIT.getThetasL();
                    thetaplus.get(layer)[i][j] += epsilon; //10.e-4
                    List<double[][]> thetasLtplus = thetasINIT.getThetasLTranspose(thetaplus);
                    AccumList.get(0).reset();
                    AccumList.get(1).reset();
                    feedpropAlg.getfeedbackprop("feed", classmode, yavg, event, thetasLtplus, thetasL, actiF, dactiF, AccumList, dthetasAccL, mu, velocity, optimization);
                    double sqwtplu = thetasINIT.getthetassqw(thetaplus); //thetas squared
                    double costplus = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtplu); //cost reg

                    //thetasMinus
                    List<double[][]> thetasmin = thetasINIT.getThetasL();
                    thetasmin.get(layer)[i][j] -= epsilon;
                    List<double[][]> thetasLtmins = thetasINIT.getThetasLTranspose(thetasmin);
                    AccumList.get(0).reset();
                    AccumList.get(1).reset();
                    feedpropAlg.getfeedbackprop("feed", classmode, yavg, event, thetasLtmins, thetasL, actiF, dactiF, AccumList, dthetasAccL, mu, velocity, optimization);
                    double sqwtmin = thetasINIT.getthetassqw(thetasmin); //thetas squared
                    double costmin = (1./AccumList.get(1).value() * AccumList.get(0).value()) + (lmbda/2*sqwtmin); //cost reg

                    //gradient check
                    gAprox.get(layer)[i][j] = (costplus-costmin) / (2*epsilon);

                }
            }
        }
        double[] gAproxV = thetasINIT.getflatThetas(gAprox);

        double gradcheck = DoubleStream.of(gAproxV).sum();
        double gthetasor = DoubleStream.of(dthetasV).sum();
        double gradientc = gradcheck-gthetasor;

        System.out.print("\nGradient Checking");
        System.out.printf("%.6f", gradientc);
        //System.out.print("gthetas backp ");
        //System.out.printf("%.6f", gthetasor);
        //System.out.print("\ngthetas check ");
        //System.out.printf("%.6f", gradcheck);
        //System.out.print("\ngthetas diffg ");
        //System.out.printf("%.6f", gradientc);
        System.out.println();

    }


}
