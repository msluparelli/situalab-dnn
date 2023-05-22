package com.situalab.dlab;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.util.DoubleAccumulator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.lang.Double.min;
import static java.lang.Math.abs;
import static java.lang.Math.log;
import static java.util.stream.Collectors.joining;
import static org.apache.spark.api.java.StorageLevels.MEMORY_ONLY_SER;


public class dnn {

    public static void main(String[] args) throws Exception {


        //get params
        String[] dnnArgs = getdDNNparams(args);
        Map<String,String> nnparams = getNNparams(dnnArgs);
        for (int m=0; m<dnnArgs.length; m++){
            String key = dnnArgs[m].split(":")[0];
            String val = dnnArgs[m].split(":")[1];
            nnparams.replace(key,val);
        }
        //classification mode
        int outputUnits = Integer.parseInt(nnparams.get("output"));
        if (outputUnits>1){
            nnparams.replace("classmode", "multi");
        }

        //dnn params
        Charset charset = Charset.forName("US-ASCII");
        String datos = nnparams.get("filepath").toString();
        String optim = nnparams.get("optimization").toString();
        String learn = nnparams.get("learning").toString();
        String nnlay = nnparams.get("nnlayers").toString().replace(",", "-");
        String muval = nnparams.get("maxmu").toString();
        String muada = nnparams.get("itert").toString();
        String winit = nnparams.get("thetasinit").toString();
        String aepre = nnparams.get("AEthetas").toString();
        String lamda = nnparams.get("lmbda").toString();
        String EPoch = nnparams.get("epochs").toString();

        String deeplabmode = nnparams.get("deeplabmode").toString();
        String directory = "";
        if (deeplabmode.equals("debug")){
            directory = "/Users/situalab/GoogleDrive/situa/operaciones/deeplab/";
        }
        String modelgen = directory+"trainingoutput/modelGeneralisation/DNNmodelOutput.csv";
        Path modelgenpath = Paths.get(modelgen); //path to save thetas
        BufferedWriter genwriter = Files.newBufferedWriter(modelgenpath, charset); // thetas writter
        genwriter.close();



        //multi trainint params
        String nninputList = nnparams.get("nnlayers");
        String[] nnV = nninputList.split("-");
        List<String> nnlayersL = new ArrayList<>();
        for (int L=0; L<nnV.length; L++){nnlayersL.add(nnV[L]);}

        String learninginputList = nnparams.get("learning");
        String[] learningV = learninginputList.split(",");
        List<String> learningL = new ArrayList<>();
        for (int L=0; L<learningV.length; L++){learningL.add(learningV[L]);}


        //training
        for (int nnt=0; nnt<nnlayersL.size(); nnt++){
            String layers = nnlayersL.get(nnt);
            nnparams.replace("nnlayers", layers);
            for (int lt=0; lt<learningL.size(); lt++){

                //spark hadoop context
                String datafile = nnparams.get("filepath");
                SparkConf conf = new SparkConf().setAppName("deeplab");
                conf.setMaster("local");
                JavaSparkContext jsc = new JavaSparkContext(conf);


                //training and validation data
                String dlab = nnparams.get("filepath").toString();
                String trainfilePath="hdfs://localhost:9000/user/"+nnparams.get("local").toString()+"/input/"+dlab+"train"; //train
                JavaRDD<LabeledPoint> trainRDD = MLUtils.loadLibSVMFile(jsc.sc(), trainfilePath).toJavaRDD();
                String valfilePath="hdfs://localhost:9000/user/"+nnparams.get("local").toString()+"/input/"+dlab+"val"; //train
                JavaRDD<LabeledPoint> valRDD = MLUtils.loadLibSVMFile(jsc.sc(), valfilePath).toJavaRDD(); //take all validation data



                //persist RDD in memory
                trainRDD.persist(MEMORY_ONLY_SER);
                valRDD.persist(MEMORY_ONLY_SER);

                String learningR = learningL.get(lt);
                nnparams.replace("learning", learningR);
                trainThetas(jsc, nnparams, trainRDD, valRDD, modelgen);

                jsc.stop();
            }
        }
    }


    //train Weights
    public static void trainThetas(JavaSparkContext jsc,
                                   Map<String,String> nnparams,
                                   JavaRDD<LabeledPoint> trainRDD,
                                   JavaRDD<LabeledPoint> valRDD,
                                   String modelgen) throws IOException {

        //Java Class Objects INIT
        dnn_activation adF = new dnn_activation(); //activation functions
        dnn_feedback feedbackProp = new dnn_feedback(); //feedbackprop
        dnn_optimization optMethod = new dnn_optimization(); //optimization
        dnn_gradientcheck gradientChecking = new dnn_gradientcheck();



        //check this when building artifact!!!!!
        String deeplabmode = nnparams.get("deeplabmode").toString();
        String directory = "";
        if (deeplabmode.equals("debug")){
            directory = "/Users/situalab/GoogleDrive/situa/operaciones/deeplab/";
        }


        //data params
        long n_train = trainRDD.count();
        long n_val = valRDD.count();



        //nnparams
        int seedN = Integer.parseInt(nnparams.get("seed")); String rndINIT = nnparams.get("rnd_modINIT").toString();
        String fan_modINIT = nnparams.get("fan_modINIT").toString(); int epochsN = Integer.parseInt(nnparams.get("epochs"));
        String nnA = nnparams.get("thetasinit").toString(); String learning_mod = nnparams.get("learning_mod").toString();
        String pretrainedthetas = directory+"deeplab_thetasPT/"+nnparams.get("pretrained").toString()+".csv";
        String AEthetaspath = directory+"weightsAE/"+nnparams.get("AEthetas").toString()+".csv";
        String printtest = nnparams.get("printtest");
        String classmode = nnparams.get("classmode").toString();


        //nnarchitecture
        String dnn = nnparams.get("nnlayers");
        int [] neuronsLayer = Stream.of(dnn.split(",")).mapToInt(Integer::parseInt).toArray();
        String dlab = nnparams.get("filepath");


        //learning & regularisation
        double learning = Double.parseDouble(nnparams.get("learning")); //learning rate, take in consideration SGD/olGD 2e.-1 o 8e.-2
        double lmbda = Double.parseDouble(nnparams.get("lmbda")); //regularisation term
        double maxmu = Double.parseDouble(nnparams.get("maxmu")); double itert = Double.parseDouble(nnparams.get("itert")); //mu threshold



        //activation function
        dnn_activation.activateV<Double> actiF = adF.sigmoid;
        dnn_activation.activateV<Double> dactiF = adF.dsigmoid;
        String activationFunction = nnparams.get("activation").toString(); //activation function: sigmoid/dsigmoid, ftanh/dtanh, ReLU/dReLU, LReLU/dLReLU, ELU/dELU
        if(activationFunction.equals("ReLU")) {actiF = adF.ReLU;dactiF = adF.dReLU; System.out.println("activation:"+activationFunction+" nnlayers:"+dnn+" learning:"+learning+" lambda:"+lmbda);}
        if(activationFunction.equals("LReLU")) {actiF = adF.LReLU;dactiF = adF.dLReLU; System.out.println("activation:"+activationFunction+" nnlayers:"+dnn+" learning:"+learning+" lambda:"+lmbda);}
        if(activationFunction.equals("ELU")) {actiF = adF.ELU;dactiF = adF.dELU; System.out.println("activation:"+activationFunction+" nnlayers:"+dnn+" learning:"+learning+" lambda:"+lmbda);}
        if(activationFunction.equals("sigmoid")) {actiF = adF.sigmoid;dactiF = adF.dsigmoid; System.out.println("activation:"+activationFunction+" nnlayers:"+dnn+" learning:"+learning+" lambda:"+lmbda);}
        if(activationFunction.equals("tanh")) {actiF = adF.ftanh;dactiF = adF.dtanh; System.out.println("activation:"+activationFunction+" nnlayers:"+dnn+" learning:"+learning+" lambda:"+lmbda);}


        //optimization and update
        String CG = nnparams.get("checkGradients").toString();
        String gradientD = nnparams.get("gradient"); String optimization = nnparams.get("optimization");


        //thetas random + deep learning architecture
        List<double[][]> thetasL = new ArrayList<double[][]>(); //thetas
        int featuresN = trainRDD.first().features().size();
        int outputUnits = Integer.parseInt(nnparams.get("output"));
        dnn_thetaspretrained thetasPT = new dnn_thetaspretrained(); //pretrained thetas
        dnn_thetas thetasINIT = new dnn_thetas(featuresN, neuronsLayer, outputUnits, seedN, rndINIT, fan_modINIT); //deeplab_thetas INIT

        List<double[][]> thetasLini = thetasINIT.getThetasL(); //thetas random INIT
        System.out.println(nnA+"thetas random/architecture size:"+thetasLini.size());

        if (nnA.equals("AE")){
            List<double[][]> thetasLpre = thetasPT.thetasListpretrain(AEthetaspath); //thetas pretrained
            for (int layer=0; layer < thetasLpre.size(); layer++){
                thetasL.add(thetasLpre.get(layer));
            }
            for (int layer=thetasL.size(); layer<thetasLini.size(); layer++){
                thetasL.add(thetasLini.get(layer));
            }
        }

        //thetasList random
        if (nnA.equals("RND")){
            thetasL = thetasLini;
        }

        if (nnA.equals("PT")){
            List<double[][]> thetasLpre = thetasPT.thetasListpretrain(pretrainedthetas); //thetas pretrained
            for (int ptlayer=0; ptlayer<thetasLpre.size()-1; ptlayer++) {
                System.out.println("thetas pt layer:" + ptlayer);
                thetasL.add(thetasLpre.get(ptlayer)); //pretrained thetas layers except last
            }
            for (int layer=thetasLpre.size()-1; layer<thetasLini.size(); layer++){
                System.out.println("thetas layer:"+layer);
                thetasL.add(thetasLini.get(layer)); //add remaining layers
            }
        }

        //thetas architecture
        for (int layer=0; layer < thetasL.size(); layer++){
            System.out.println("thetas:"+layer+" "+thetasL.get(layer).length+" "+thetasL.get(layer)[0].length);
        }
        System.out.println("thetasL size:"+thetasL.size());




        //thetas params
        double[] thetastrained = thetasINIT.getflatThetas(thetasL); //to flaten thetasL
        double[] lastthetastrained = thetasINIT.getflatThetas(thetasL); //last thetas trained
        List<dnn_thetasAccumM> dthetasAcc = new ArrayList<dnn_thetasAccumM>(); //derivative thetas, Accumulator
        List<dnn_thetasAccumM> dHdthetasAcc = new ArrayList<dnn_thetasAccumM>(); //derivative thetas, Accumulator
        List<double[][]> gthetas = new ArrayList<double[][]>(); //gradient thetas
        List<double[][]> gHdthetas = new ArrayList<double[][]>(); //gradient thetas to compute Hessiand
        List<double[][]> gthetasAcc = thetasINIT.getThetasLZeros(thetasL); //gradiente accumulator
        List<double[][]> vWA = thetasINIT.getThetasLZeros(thetasL); //velocity accumulator

        //SGD params
        Random sgdSEED = new Random();
        sgdSEED.setSeed(2332);
        sgdSEED.ints(epochsN);
        int[] sgsSEEDvalues = new int[epochsN];
        for (int s=0; s<epochsN; s++){sgsSEEDvalues[s] = sgdSEED.nextInt(9999);}
        double sgd = min(512./512., 512./trainRDD.count()); double gross = 1.0-sgd; double[] wtSplit = {sgd, gross};long seedSplit = 2332;


        //gradientChecking
        if(CG.equals("si")){gradientChecking.checkGradient(trainRDD.first());}


        //write cost and thetas
        Charset charset = Charset.forName("US-ASCII");
        String nnlayers = Arrays.toString(neuronsLayer).replaceAll("\\s+","");
        String costpathfile = directory+"trainingoutput/modelCosts/"+dlab+"_"+featuresN+"_"+nnlayers+"_"+activationFunction+"_"+gradientD+"_"+optimization+"_"+String.valueOf(learning)+".csv";
        String nnthetaspath = directory+"weights/"+dlab+"_"+nnlayers+"_"+String.valueOf(learning)+"_"+Integer.toString(epochsN)+".csv";
        double breakthreshold=1.e-6; //cost validation
        int[] feats = {featuresN};
        int[] outpv = {outputUnits};
        int[] thetasArchitecture = IntStream.concat(Arrays.stream(feats), IntStream.concat(Arrays.stream(neuronsLayer), Arrays.stream(outpv))).toArray();
        String thetasArchF ="";
        for (int a=0; a<thetasArchitecture.length; a++){
            thetasArchF+= (Integer.toString(thetasArchitecture[a]))+"_";
        }
        thetasArchF = thetasArchF.substring(0, thetasArchF.length()-1)+"_"+activationFunction+"\n";
        if (deeplabmode.equals("save")){

            //cost writer
            Path nncost = Paths.get(costpathfile); //path to save costs
            BufferedWriter costwriter = Files.newBufferedWriter(nncost, charset); //cost file writter
            String filelabel = "epoch,cost_train,cost_val\n"; //encabezado cost file
            costwriter.write(filelabel); //grabar encabezado
            costwriter.close();

            Path nnthetas = Paths.get(nnthetaspath); //path to save thetas
            BufferedWriter thetaswriter = Files.newBufferedWriter(nnthetas, charset); // thetas writter
            thetaswriter.write(thetasArchF);
            thetaswriter.close();

        }



        //evaluation params
        double best_cost_val = 2.0; //best cost validation
        int best_cost_val_epoch = 0;
        //int best_acc_val_epoch = 0;


        //training iteration
        System.out.println("\ntraining iterations:"+epochsN+"\n");
        for (int epoch=0; epoch<epochsN; epoch++){

            //feed propagation params
            Broadcast<List<double[][]>> thetasLbd = jsc.broadcast(thetasL); //broadcast deeplab_thetas
            List<double[][]> thetasLt = thetasINIT.getThetasLTranspose(thetasL); //weigths transposed
            Broadcast<List<double[][]>> thetasLtBD = jsc.broadcast(thetasLt); //broadcast weights Arrays transposed

            //backpropagation params
            dthetasAcc = thetasINIT.getAccThetas(thetasL); //derivative
            for (int a=0; a<dthetasAcc.size(); a++){
                String wName = "dW".concat(Integer.toString(a));
                jsc.sc().register(dthetasAcc.get(a), wName); //register accumulators List
            }
            dHdthetasAcc = thetasINIT.getAccThetas(thetasL); //derivative to compute Hessiand
            for (int a=0; a<dHdthetasAcc.size(); a++){
                String wName = "dHdW".concat(Integer.toString(a));
                jsc.sc().register(dHdthetasAcc.get(a), wName); //register derivative accumulators List
            }
            //final variables for iteration
            dnn_activation.activateV<Double> actiFfinal = actiF;
            dnn_activation.activateV<Double> dactiFfinal = dactiF;
            List<DoubleAccumulator> AccumListFeed = initAccumulators(jsc); //accumulators
            List<DoubleAccumulator> AccumListBack = initAccumulators(jsc); //accumulators
            List<DoubleAccumulator> AccumListHdBack = initAccumulators(jsc); //accumulators
            List<dnn_thetasAccumM> dthetasAccL = dthetasAcc; //final variable
            List<dnn_thetasAccumM> dHdthetasAccL = dHdthetasAcc; //final variable
            double yavg = 0; //no se usa en feed / back / val
            double e = -1-(log(((epoch/itert)+1))/log(2));
            double mu = Math.min(1-Math.pow(2,e), maxmu); //inspired by nesterov momentum value (in Sutskever)
            double mu_n = min(1-3/(epoch+5), maxmu); //nesterov momentum value (in Sutskever)
            List<double[][]> velocity = vWA;
            NumberFormat formatter = new DecimalFormat("#0.000");



            //train, feed propagarion
            trainRDD.foreach(xrdd -> feedbackProp.getfeedbackprop("feed", classmode, yavg, xrdd, thetasLtBD.value(), thetasLbd.value(), actiFfinal, dactiFfinal, AccumListFeed, dthetasAccL, mu, velocity, optimization));


            //backpropagation, SGD o olGD
            JavaRDD<LabeledPoint> sgdRDD = trainRDD.randomSplit(wtSplit, sgsSEEDvalues[epoch])[0]; //split and select data
            List<LabeledPoint> olgdRDD = trainRDD.takeSample(true,1, epoch+10);
            if (gradientD=="SGD"){
                sgdRDD.foreach(xrdd -> feedbackProp.getfeedbackprop("back", classmode, yavg, xrdd, thetasLtBD.value(), thetasLbd.value(), actiFfinal, dactiFfinal, AccumListBack, dthetasAccL, mu, velocity, optimization));
            }
            if (gradientD=="olGD"){
                feedbackProp.getfeedbackprop("back", classmode, yavg, olgdRDD.get(0), thetasLtBD.value(), thetasLbd.value(), actiF, dactiF, AccumListBack, dthetasAccL, mu, velocity, optimization);
            }


            String optimizationHF="noHF";
            if (optimizationHF.equals("HF")){
                gthetas = optMethod.getderivativeBackProp(thetasLbd.value(), dthetasAccL, lmbda, AccumListBack.get(1).value()); //derivative, gradient
                List<double[][]> thetaseL = new ArrayList<>(); //Hessian d optimization
                thetasINIT.getthetasepsilon(thetasL, gthetas, thetaseL);
                Broadcast<List<double[][]>> thetaseLbd = jsc.broadcast(thetaseL); //broadcast deeplab_thetas
                List<double[][]> thetaseLt = thetasINIT.getThetasLTranspose(thetaseL); //weigths transposed
                Broadcast<List<double[][]>> thetaseLtBD = jsc.broadcast(thetaseLt); //broadcast weights Arrays transposed
                sgdRDD.foreach(xrdd -> feedbackProp.getfeedbackprop("back", classmode, yavg, xrdd, thetaseLtBD.value(), thetaseLbd.value(), actiFfinal, dactiFfinal, AccumListHdBack, dHdthetasAccL, mu, velocity, optimization));
                gHdthetas = optMethod.getderivativeBackProp(thetasLbd.value(), dHdthetasAccL, lmbda, AccumListHdBack.get(1).value()); //derivative, gradients to cumpute Hessian d
                List<double[][]> Bhessiand = new ArrayList<>(); //Hessian d optimization
                optMethod.getBHessiand(Bhessiand, gHdthetas, gthetas, lmbda);
            }


            //optimization
            double n = trainRDD.count();
            if (optimization.equals("vanilla")){
                gthetas = optMethod.getderivativeBackProp(thetasLbd.value(), dthetasAccL, lmbda, AccumListBack.get(1).value()); //derivative, gradients
                gthetasAcc = optMethod.accumGradients(gthetas, gthetasAcc); //accumulate gradients for ADAGRAD
                thetasL = optMethod.getthetasVanillaUpdated(thetasLbd.value(), gthetas, learning, gthetasAcc, AccumListBack.get(1).value(), lmbda, n); //vanillaUpdate, update
            }
            if (optimization.equals("momentum") | optimization.equals("nesterov")){
                gthetas = optMethod.getderivativeBackProp(thetasLbd.value(), dthetasAccL, lmbda, AccumListBack.get(1).value()); //derivative, gradients
                gthetasAcc = optMethod.accumGradients(gthetas, gthetasAcc); //accumulate gradients for ADAGRAD
                vWA = optMethod.getmuVelocity(gthetas, vWA, mu, learning, gthetasAcc, learning_mod); //velocity
                thetasL = optMethod.getthetasMuUpdated(thetasLbd.value(), vWA); //update
            }


            //validation
            List<double[][]> thetasLtVAL = thetasINIT.getThetasLTranspose(thetasL); //updated deeplab_thetas transposed
            Broadcast<List<double[][]>> thetasLtBDVAL = jsc.broadcast(thetasLtVAL); //broadcast updated deeplab_thetas transposed
            List<DoubleAccumulator> AccumListVali = initAccumulators(jsc); //accumulators
            valRDD.foreach(xrdd -> feedbackProp.getfeedbackprop("val", classmode, yavg, xrdd, thetasLtBDVAL.value(), thetasLbd.value(), actiFfinal, dactiFfinal, AccumListVali, dthetasAccL, mu, velocity, optimization));


            //print cost function output
            double regterm = lmbda/2;
            double sqwt = thetasINIT.getthetassqw(thetasL); //W squared

            //cost train
            double cost = 1./AccumListFeed.get(1).value() * AccumListFeed.get(0).value(); //train cost
            double regCostTrain = cost + (regterm*sqwt); //train reg cost

            //cost val
            double costVal = 1./AccumListVali.get(1).value() * AccumListVali.get(0).value(); //val cost
            double regCostVAL = costVal + (regterm*sqwt); //VAL reg cost
            double costdiff = abs(regCostVAL-best_cost_val);
            if (regCostVAL<=best_cost_val) {
                best_cost_val = regCostVAL;
                best_cost_val_epoch = epoch+1;
            }
            double bnAccuracyval = AccumListVali.get(2).value() / valRDD.count();


            String regcost = Double.toString(epoch)+","+Double.toString(regCostTrain)+","+Double.toString(regCostVAL)+"\n";
            if (deeplabmode.equals("save")){
                Writer costoutput = new BufferedWriter(new FileWriter(costpathfile, true));
                costoutput.append(regcost);
                costoutput.close();
            }




            String accval_ = formatter.format(bnAccuracyval);
            String best_cost_val_ = formatter.format(best_cost_val);
            String regcostval_ = formatter.format(regCostVAL);
            System.out.print("\repoch "+(epoch+1)+" events ");
            System.out.print("val cost:"+regcostval_+"[");
            System.out.print(best_cost_val_+" "+best_cost_val_epoch+"] ");
            System.out.print("val acc:"+accval_+" ");
            System.out.print("training count: "+n_train+" validation count: "+n_val);




            //trained thetas
            if (costdiff<=breakthreshold) {
                thetastrained = thetasINIT.getflatThetas(thetasL);
            }



            //end training
            if (epoch==(epochsN-1)*1){
                System.out.println("\n\nend training, saving output... ");

                //thetas trained
                int usethetas = 1; //0 bestthetastrained; 1 lasthetastrained

                List<double[]> thetasV = new ArrayList<double[]>();
                lastthetastrained = thetasINIT.getflatThetas(thetasL); //last thetas trained
                thetasV.add(thetastrained);
                thetasV.add(lastthetastrained);

                //save thetas
                for (int t=0; t<thetasV.get(usethetas).length; t++){
                    String theta = thetasV.get(usethetas)[t]+"\n";

                    if (deeplabmode.equals("save")){
                        Writer thetaoutput = new BufferedWriter(new FileWriter(nnthetaspath, true));
                        thetaoutput.append(theta);
                        thetaoutput.close();
                    }
                }
                System.out.println("\nthetas saved mod("+usethetas+") 0=BESTthetastrained 1=LASTthetastrained");

                //write model generalisation
                String bestcostval;bestcostval = formatter.format(best_cost_val);
                String nnP = nnparams.entrySet().stream().map(Object::toString).collect(joining(" ")).replace(",", "_");
                String datos = nnparams.get("filepath").toString();
                String optim = nnparams.get("optimization").toString();
                String learn = nnparams.get("learning").toString();
                String nnlay = nnparams.get("nnlayers").toString().replace(",", "-");
                String muval = nnparams.get("maxmu").toString();
                String muada = nnparams.get("itert").toString();
                String winit = nnparams.get("thetasinit").toString();
                String aepre = nnparams.get("AEthetas").toString();
                String pthet = nnparams.get("pretrained").toString();
                String lamda = nnparams.get("lmbda").toString();
                String EPoch = nnparams.get("epochs").toString();


                String ar = nnlay+" "+EPoch+" "+learn;
                String op = optim+"_"+muval+"_"+muada;
                String re = winit+"_"+aepre+pthet+" "+lamda;


                int hlayers = thetasL.size()-1;
                String modelgeneralisation = datos+";"+ar+";"+hlayers+";"+best_cost_val_epoch+";"+bestcostval.replace(".",",")+";"+op+";"+re+"\n";


                if (deeplabmode.equals("save")){
                    try (Writer modelWriter = new FileWriter(modelgen, true)){
                        modelWriter.write(modelgeneralisation);
                    } catch (IOException error) {
                        System.out.println("Problem occurs when deleting the directory : " + modelgen);
                        error.printStackTrace();
                    }
                }

            }

            thetasLbd.destroy(); //destroy broadcast at the end of loop
            thetasLtBD.destroy(); //destroy broadcast at the end of loop
            thetasLtBDVAL.destroy(); //destroy broadcast at the end of loop

        }


        System.out.println("\n"+directory+costpathfile);
        //System.out.println(nnthetaspath);
        //System.out.println(nnparams);



    }





    //Accumulators
    public static List<DoubleAccumulator> initAccumulators(JavaSparkContext jsc) {

        DoubleAccumulator JthetasCost = new DoubleAccumulator(); //J Accumulator
        DoubleAccumulator mEvents = new DoubleAccumulator(); //events Accumulator
        DoubleAccumulator binA = new DoubleAccumulator(); //binary Accumulator
        DoubleAccumulator sqeA = new DoubleAccumulator(); //sqe Accumulator
        DoubleAccumulator avgA = new DoubleAccumulator(); //average error Accumulator
        List<DoubleAccumulator> AccumList = new ArrayList<>();
        JthetasCost = jsc.sc().doubleAccumulator("cost function"); //init at zero
        mEvents = jsc.sc().doubleAccumulator("events"); //init at zero
        binA = jsc.sc().doubleAccumulator("bin"); //init at zero
        sqeA = jsc.sc().doubleAccumulator("sqe"); //init at zero
        avgA = jsc.sc().doubleAccumulator("avg"); //init at zero
        AccumList.add(JthetasCost);
        AccumList.add(mEvents);
        AccumList.add(binA);
        AccumList.add(sqeA);
        AccumList.add(avgA);

        return AccumList;
    }










    //get params
    private static String[] getdDNNparams(String[] args){
        String deeplabDNNparams;
        try{
            deeplabDNNparams = args[0];
        } catch (java.lang.ArrayIndexOutOfBoundsException e) {
            deeplabDNNparams = "filepath:skitagGPSzs;val:si;epochs:10;activation:ReLU;output:2;learning:1.e-1;nnlayers:40;thetasinit:AE;AEthetas:skitagGPSzs[40]AE;deeplabmode:debug";
        }
        return deeplabDNNparams.split(";");
    }
    private static Map<String,String> getNNparams(String[] args){

        Map<String,String> nnparams = new HashMap<>();
        nnparams.put("filepath", "");
        nnparams.put("seed", "4237842");
        nnparams.put("rnd_modINIT", "xnormal");
        nnparams.put("fan_modINIT", "fan_in");
        nnparams.put("nnlayers", "20");
        nnparams.put("output", "1"); //default:1:binary classification
        nnparams.put("classmode", "bin"); //default:1:binary classification // bin o multi
        nnparams.put("epochs", "200");
        nnparams.put("itert", "1");
        nnparams.put("thetasinit", "RND");
        nnparams.put("pretrained", "");
        nnparams.put("AEthetas", "");
        nnparams.put("printtest", "yes");
        nnparams.put("learning", "1.e-1");
        nnparams.put("learning_mod", "");
        nnparams.put("lmbda", "1.e-10");
        nnparams.put("maxmu", "0.95");
        nnparams.put("activation", "sigmoid");
        nnparams.put("optimization", "momentum");
        nnparams.put("gradient", "SGD");
        nnparams.put("checkGradients", "no");
        nnparams.put("local", "situalab");
        nnparams.put("deeplabmode", "");

        //save: save cost and thetas;
        //debug: debug
        //"": train;
        //"modelOutput" to save in modelOutput file



        return nnparams;
    }

}
