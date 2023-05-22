package com.situalab.dlab;

import org.apache.commons.lang.ArrayUtils;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.util.DoubleAccumulator;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.StrictMath.pow;


public class dnn_feedback implements Serializable {

    //attributes, with iteration

    //constructor, with iteration

    //method
    public void getfeedbackprop(String prop,
                                String classmode,
                                double yavg,
                                LabeledPoint xvalues,
                                List<double[][]> thetasLt,
                                List<double[][]> thetasL,
                                dnn_activation.activateV<Double> activation,
                                dnn_activation.activateV<Double> dactivation,
                                List<DoubleAccumulator> AccumList, //events y costs
                                List<dnn_thetasAccumM> dthetasAccL, //derivatives
                                double mu,
                                List<double[][]> velocity,
                                String optimization
    ) {


        //params
        int hiddenL = thetasL.size()-1;

        //List of A and Z
        List<double[]> A = new ArrayList<double[]>();
        List<double[]> Z = new ArrayList<double[]>();

        //bias unit
        double[] biasUnit = {1};
        double[] zi;

        //activation input
        double[] xj = xvalues.features().toArray();
        double [] ai = ArrayUtils.addAll(biasUnit, xj);
        if (prop.equals("back")) A.add(ai); //activation List, only with backprop


        //hidden layers
        for (int ly = 0; ly<hiddenL; ly++){
            zi = getZivalues(thetasLt.get(ly), ai);
            ai = Arrays.stream(zi).map(xoz -> activation.apply(xoz)).toArray(); //activation
            ai = ArrayUtils.addAll(biasUnit, ai);

            if (prop.equals("back")) Z.add(zi); //z List, only with backprop
            if (prop.equals("back")) A.add(ai); //activation List, only with backprop
        }

        //output unit, default:binary classification
        double[] zout = getZivalues(thetasLt.get(hiddenL), ai);
        double[] aout = Arrays.stream(zout).map(xoz -> dnn_activation.sigmoid.apply(xoz)).toArray(); //activation, always sigmoid
        double[] pred = ArrayUtils.addAll(biasUnit, aout);
        double[] predR = Arrays.stream(aout).map(p -> dnn_activation.predres.apply(p)).toArray(); //residual values logloss function

        //output unit, multi classification
        if (classmode.equals("multi")){
            double[] expzj = Arrays.stream(zout).map(xoz -> dnn_activation.expV.apply(xoz)).toArray();
            double Eexpzj = Arrays.stream(expzj).sum();
            aout = Arrays.stream(expzj).map(zj -> zj/Eexpzj).toArray();
        }



        //update Z & A list
        if (prop.equals("back")) Z.add(zout); //z List, only with backprop
        if (prop.equals("back")) A.add(aout); //activation List, only with backprop


        //y value and cost
        double[] y = new double[aout.length];
        for (int i=0; i<y.length; i++){
            y[i] = xvalues.label();
        }
        int indxclass = 0;
        if (classmode.equals("multi")){
            indxclass = (int)xvalues.label(); //correct clasification or index of classifitation
            y[indxclass] = 1;
        }

        //loss function, default:binary classification
        double[] eout = new double[y.length]; //error output
        double[] Jout = new double[y.length]; //cost value

        if (classmode.equals("linear")){

            //linear classification
            //double dpred = (1.0 / (1.0 + exp(-zout[i]))) * (1 - (1.0 / (1.0 + exp(-zout[i]))));
            //double daout = aout[i]*(1-aout[i]); //stanford UFLD
            //eout[i] = (pred[(i+1)]-y[(i)])*daout; //-(y[i]-pred[i+1])*dpred o (pred[i+1]-y[i])*dpred, linear classification
            //Jout[i] = (pow((y[i]-aout[(i)]),2))/2; //stanford UFLDL, squared error, linear classification

        }

        if (classmode.equals("bin")){
            for (int i=0; i<y.length; i++){

                //binary classification
                eout[i] = (aout[(i)]-y[i]); //verificado, utilizado con logloss
                Jout[i] = -(y[i]*log(predR[i]) + (1-y[i])*log(1-predR[i])); //sklearn, cross entropy, https://en.wikipedia.org/wiki/Cross_entropy

            }
            AccumList.get(0).add(Jout[0]); //eJou+Jout, lineal classification; elog+logl binary classification



            if (prop.equals("test") | prop=="val"){
                //errors, accuracy
                int bnpred = 0;
                if (aout[0]>=0.5) bnpred = 1;
                if (bnpred==y[0]) AccumList.get(2).add(1.0);
                double sqe = Math.pow((y[0]-aout[0]),2);
                AccumList.get(3).add(sqe);
                double avge = Math.pow((y[0]-yavg),2);
                AccumList.get(4).add(avge);
            }

        }

        if (classmode.equals("multi")){
            //LossFunction (multiclasification)
            double Jout_ = -log(aout[indxclass]);
            AccumList.get(0).add(Jout_); //eJou+Jout, lineal classification; elog+logl binary classification


            //Output layer delta(error)
            eout = aout.clone();
            eout[indxclass] -=1;

            //Model performance
            double maxpred = Arrays.stream(aout).max().getAsDouble();
            int indxpred = 0;
            for (int k=0; k<aout.length; k++){
                if (aout[k]==maxpred) {
                    indxpred+=k;
                    break;
                }
            }

            if (prop.equals("test")){
                //System.out.println(Arrays.toString(aout));
                //System.out.println(Arrays.toString(xj));
            }


            //prediction output, (prop=="test" | prop=="val") o (prop=="test")
            if (prop.equals("test") | prop=="val"){
                if (indxpred==indxclass) AccumList.get(2).add(1.0);
            }
        }








        //backpropagation FUN algoritm
        dnn_optimization optM = new dnn_optimization();
        List<double[][]> thetasLn = new ArrayList<>();
        if(optimization.equals("nesterov")){
            thetasLn = optM.getthetasL_nesterov(thetasL, velocity, mu);
        }

        if(prop.equals("back") & optimization.equals("nesterov")){
            getBackPropagation(eout, A, Z, thetasLn, dthetasAccL, dactivation); //back propagation
        }
        if(prop.equals("back")){
            getBackPropagation(eout, A, Z, thetasL, dthetasAccL, dactivation); //back propagation
        }
        AccumList.get(1).add(1.0); //count SGD events

    }



    //get zi values (matrix, vector multplication)
    public double[] getZivalues(double[][] M, double[] v){
        double[] Mv = new double[M.length];
        for (int row=0; row<M.length; row++){
            double rowSum = 0;
            for (int col=0; col<M[row].length; col++){
                rowSum += (M[row][col]*v[col]);
            }
            Mv[row] = rowSum;
        }
        return Mv;
    }

    //backpropagation algorithm
    public void getBackPropagation(double[] ej,
                                   List<double[]> A,
                                   List<double[]> Z,
                                   List<double[][]> thetasL,
                                   List<dnn_thetasAccumM> dthetasAccL,
                                   dnn_activation.activateV<Double> dactivation){

        int hiddenL = thetasL.size()-1; //hidden layers
        double[] biasUnit = {1}; //bias unit

        //backPropagation output layer, verificado esta OK
        double[] aj = A.get(hiddenL); //vector
        double [][] dthetaoutputlayer = new double[aj.length][ej.length]; //init dtheta matrix
        getderivatevethetaslayer(dthetaoutputlayer, aj, ej); //get dWl(hiddenL), verificado
        dthetasAccL.get(hiddenL).add(dthetaoutputlayer); //add dWl to W Accumulator

        //backpropafation hiddenL, verificado esta OK, layers 1,0
        for (int bplayer=hiddenL-1; bplayer>=0; bplayer--){

            //get deactivation values
            double[] dziBP = Arrays.stream(Z.get(bplayer)).map(xoz -> dactivation.apply(xoz)).toArray(); //get dactivation z
            double[] dzi = Arrays.stream(A.get(bplayer+1)).map(a -> a*(1-a)).toArray();
            dziBP = ArrayUtils.addAll(biasUnit, dziBP); //incluye bias unit

            //get errores
            double[] ejlayer = new double[(thetasL.get(bplayer)[0].length+1)]; //zero vector para calcular ejl (thetas col+1 o thetasT row+1)
            getEjLayer(ejlayer, thetasL.get(bplayer+1), ej, dziBP); //get ejl{w(l+1), ej(l+1), z(l)}
            ej = Arrays.copyOfRange(ejlayer,1, ejlayer.length); // ej(l)

            //get derivative
            aj = A.get(bplayer); //incluye bias unit
            double[][] dthetahiddenlayer = new double[aj.length][ej.length]; //init dtheta matrix
            getderivatevethetaslayer(dthetahiddenlayer, aj, ej); //get dWl, partial derivatives
            dthetasAccL.get(bplayer).add(dthetahiddenlayer); //add dWl to W Accumulator

        }
    }


    //get dthetaslayer
    public void getderivatevethetaslayer(double[][] dthetas,
                                         double[] aj,
                                         double[] ej){
        for (int row = 0; row < aj.length; row++) {
            for (int col = 0; col < ej.length; col++) {
                dthetas[row][col] = aj[row]*ej[col];
            }
        }
    }



    //get ej
    public void getEjLayer(double[] ejout, double[][] Wij, double[] ej, double[] dzi) {
        for (int row = 0; row < Wij.length; row++) {
            double rowSum = 0;
            for (int col = 0; col < Wij[row].length; col++) {
                rowSum += (Wij[row][col]*ej[col]);
            }
            ejout[row] = rowSum*dzi[row];
        }
    }

    //get thetas
    public List<double[][]> getthetasepsilonZERO (List<double[][]> thetasL){
        List<double[][]> thetaseL = new ArrayList<>();
        for (int layer=0; layer<thetasL.size(); layer++){
            int row = thetasL.get(layer).length;
            int col = thetasL.get(layer)[0].length;
            double[][] thetase = new double[row][col];
            thetaseL.add(thetase);
        }
        return thetaseL;

    }

    //clipping gradients
    public double[] clippingGradient(double[] gradient){

        double[] clippedg = gradient;

        double gnorm = Math.sqrt(Arrays.stream(gradient).map(g -> Math.pow(g,2)).sum());
        double v = 1;
        if (gnorm > v){
            clippedg = Arrays.stream(gradient).map(g -> (g*v)/gnorm).toArray();
        }

        return clippedg;

    }




}
