/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ---------------------------------------------------------------------
 */

package org.numenta.nupic.examples.sp;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;

import mnist.MNISTViewer;
import mnist.MnistManager;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.SparseInstance;
import net.sf.javaml.distance.DistanceMeasure;
import net.sf.javaml.distance.EuclideanDistance;
import net.sf.javaml.distance.ManhattanDistance;

import org.numenta.nupic.Connections;
import org.numenta.nupic.Parameters;
import org.numenta.nupic.Parameters.KEY;
import org.numenta.nupic.research.SpatialPooler;
import org.numenta.nupic.util.ArrayUtils;
import org.numenta.nupic.util.Condition;

/**
 * A simple program that demonstrates the working of the spatial pooler
 * 
 * @author Neal Miller
 */
public class RunMNIST {
    private SpatialPooler sp;
    private Parameters parameters;
    private Connections mem;
    private int[] inputArray;
    private int[] activeArray;
    private int inputSize;
    private int columnNumber;
    
    /**
     * 
     * @param inputDimensions         The size of the input.  {m, n} will give a size of m x n
     * @param columnDimensions        The size of the 2 dimensional array of columns
     */
    RunMNIST(int[] inputDimensions, int[] columnDimensions) {
        inputSize = 1;
        columnNumber = 1;
        for (int x : inputDimensions) {
            inputSize *= x;
        }
        for (int x : columnDimensions) {
            columnNumber *= x;
        }
        activeArray = new int[columnNumber];
        inputArray = new int[inputSize];
        
        parameters = Parameters.getSpatialDefaultParameters();
        parameters.setParameterByKey(KEY.INPUT_DIMENSIONS, inputDimensions);
        parameters.setParameterByKey(KEY.COLUMN_DIMENSIONS, columnDimensions);
        parameters.setParameterByKey(KEY.POTENTIAL_RADIUS, inputSize);
        parameters.setParameterByKey(KEY.POTENTIAL_PCT, .9);
        parameters.setParameterByKey(KEY.GLOBAL_INHIBITIONS, true);
        parameters.setParameterByKey(KEY.NUM_ACTIVE_COLUMNS_PER_INH_AREA, 240.0);
        parameters.setParameterByKey(KEY.SYN_PERM_ACTIVE_INC, 0.00);
        parameters.setParameterByKey(KEY.SYN_PERM_INACTIVE_DEC, 0.00);
        parameters.setParameterByKey(KEY.SYN_PERM_CONNECTED, 0.2);
        parameters.setParameterByKey(KEY.SYN_PERM_TRIM_THRESHOLD, 0.005);
        parameters.setParameterByKey(KEY.MAX_BOOST, 1.0);

        sp = new SpatialPooler();
        mem = new Connections();
        parameters.apply(mem);
        sp.init(mem);
    }
    
    /**
     * Set the current image from the MNIST reader as the inputArray
     * @throws IOException 
     */
    public void setInputArray(MnistManager manager) throws IOException {
    	int[][] image=manager.readImage();
    	graytoBW(image);
    	for (int i = 0; i < image.length; i++) {
			for (int j = 0; j < image[0].length; j++) {
				inputArray[i*image[0].length+j]=image[i][j];
			}
		}
    }
    /**
     * converts a grayscale image to black and white 
     * 		current conversion method if grayscale value is more than half
     * 			then black
     * 			else white
     * @param grayImage
     */
    private void graytoBW(int[][] grayImage) {
    	for (int i = 0; i < grayImage.length; i++) {
			for (int j = 0; j < grayImage[0].length; j++) {
				grayImage[i][j]=grayImage[i][j]>127?0:1;
			}
		}
    }
    /**
     * Create a random input vector
     */
    public void createInput() {
        for (int i = 0; i < 70; i++) System.out.print("-");
        System.out.print("Creating a random input vector");
        for (int i = 0; i < 70; i++) System.out.print("-");
        System.out.println();
        
        inputArray = new int[inputSize];
        
        Random rand = new Random();
        for (int i = 0; i < inputSize; i++) {
            // nextInt(2) returns 0 or 1
            inputArray[i] = rand.nextInt(2);
        }
    }
    
    /**
     * Run the spatial pooler with the input vector
     * @return int[] array of indices of all the active columns
     */
    public int[] run() {
        //for (int i = 0; i < 80; i++) System.out.print("-");
        //System.out.print("Computing the SDR");
        //for (int i = 0; i < 70; i++) System.out.print("-");
        //System.out.println();
        
        sp.compute(mem, inputArray, activeArray, true, true);
        
        int[] res = ArrayUtils.where(activeArray, new Condition.Adapter<Object>() {
            public boolean eval(int n) {
                return n > 0;
            }
        });
        //System.out.println(Arrays.toString(res));
        return activeArray;
    }

    /**
     * Flip the value of a fraction of input bits (add noise)
     * @param noiseLevel        The percentage of total input bits that should be flipped
     */
    public void addNoise(double noiseLevel) {
        Random rand = new Random();
        for (int i = 0; i < noiseLevel*inputSize; i++) {
            int randomPosition = rand.nextInt(inputSize);
            // Flipping the bit at the randomly picked position
            inputArray[randomPosition] = 1 - inputArray[randomPosition];
        }
    }
    
    /**
     * Cacluates the centroid of the dataset
     * @param dataset
     * @return
     */
    public static double[] centroid(Dataset dataset,boolean isRealSpace){
    	double[] centroid=new double[dataset.get(1).values().size()];
    	for (int i = 0; i < centroid.length; i++) {
    		double dimCentroid=0;
    		for (Iterator iterator = dataset.iterator(); iterator.hasNext();) {
    			Instance instance = (Instance) iterator.next();
    			dimCentroid+=instance.value(i);
    		}	
    		centroid[i]=dimCentroid/dataset.size();
    		if(!isRealSpace)
    			centroid[i]=centroid[i]<Math.floor(centroid[i])+.5?Math.floor(centroid[i]):Math.ceil(centroid[i]);
		}
    	return centroid;
    }

    /**
     * Calculates the mean distance of all the points in the dataset from the centroid
     * @param centroid
     * @param dataset
     * @return
     */
    public static double meanDistance(double[] centroid, Dataset dataset,DistanceMeasure dm){
    	double mean=0;
    	for (Iterator iterator = dataset.iterator(); iterator.hasNext();) {
			Instance instance = (Instance) iterator.next();
			mean+=dm.measure(instance, new DenseInstance(centroid));
		}
    	return mean/dataset.size();
    }
    
    /**
     * Calculates the variance of distance of all the points in the dataset from the centroid
     * @param centroid
     * @param dataset
     * @return
     */
    public static double variance(double[] centroid, Dataset dataset,DistanceMeasure dm){
    	double mean=meanDistance(centroid, dataset, dm);
    	double variance=0;
    	for (Iterator iterator = dataset.iterator(); iterator.hasNext();) {
			Instance instance = (Instance) iterator.next();
			variance+=Math.pow(mean-dm.measure(instance, new DenseInstance(centroid)),2);
		}
    	return variance/dataset.size();
    }
    
    /**
     * 
     * @param args
     * @throws IOException
     */
    public static void main(String args[]) throws IOException {
//        MnistManager mnist=new MnistManager(args[0], args[1]);
    	int columns=(int) Math.pow(Integer.parseInt(args[1]), 2);
        MnistManager mnist=new MnistManager("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        RunMNIST example = new RunMNIST(new int[]{mnist.readImage().length, mnist.readImage()[0].length}, new int[]{ (int) Math.sqrt(columns),(int) Math.sqrt(columns)});
//        MNISTViewer mnistViewer=new MNISTViewer(mnist);


        DefaultDataset dataset=new DefaultDataset();
        DefaultDataset realCluster[]=new DefaultDataset[10];
        for (int i = 0; i < realCluster.length; i++) {
        	realCluster[i]=new DefaultDataset();
		}
        //run for given number of images from MNIST
        int j=0;
        for (j = 0; j < Integer.parseInt(args[0]); j++) {
            mnist.setCurrent(j+1);
            example.setInputArray(mnist);
            int output[]=example.run();
            dataset.add(new DenseInstance(ArrayUtils.toDoubleArray(output),mnist.readLabel()));
            realCluster[mnist.readLabel()].add(new DenseInstance(ArrayUtils.toDoubleArray(output),mnist.readLabel()));
            System.out.println( "Clustered "+ j);
        }
        double[][] centroids=new double[10][columns];
        for (int i = 0; i < realCluster.length; i++) {
        	centroids[i]=centroid(realCluster[i], false);
			System.out.println("Real Cluster "+i);
			System.out.println("Mean Distance "+meanDistance(centroids[i], dataset, new EuclideanDistance()));
			System.out.println("Variance "+variance(centroids[i], dataset, new EuclideanDistance()));
			System.out.println();
		}

        System.out.println("Real Cluster Mutual Distances");
        for (int i = 0; i < centroids.length; i++) {
			for (int k = 0; k < centroids.length; k++) {
				System.out.print(new EuclideanDistance().calculateDistance(new DenseInstance(centroids[i]),new DenseInstance(centroids[k]))+",");
			}
			System.out.println();
		}
        /* 
        System.out.println("Clustering");
        Dataset[] tenCluster=(new KMeans(10,1,new ManhattanDistance())).cluster(dataset);
        System.out.println("clustered");
        for (int i = 0; i < tenCluster.length; i++) {
        	int instances[]=new int[10];
        	for (Iterator iterator = tenCluster[i].iterator(); iterator.hasNext();) {
				Instance instance = (Instance) iterator.next();
				instances[(int)instance.classValue()]++;
			}
        	System.out.println("Cluster "+i);
        	for (int k = 0; k < instances.length; k++) {
        		System.out.println("Instances of "+k+ ":" +instances[k]);
        	}
        	System.out.println();
		}
        
        mnist.setCurrent(j++ + 1);
        example.setInputArray(mnist);
        int l1=mnist.readLabel();
        int f1[]=example.run();
        mnist.setCurrent(j++ + 1);
        example.setInputArray(mnist);
        int l2=mnist.readLabel();
        int f2[]=example.run();
        
        System.out.println("L1="+l1);
        System.out.println("Columns active="+ArrayUtils.where(f1, new Condition.Adapter<Object>() {
            public boolean eval(int n) {
                return n > 0;
            }
        }).length);
        System.out.println("L2="+l2);
        System.out.println("Columns active="+ArrayUtils.where(f2, new Condition.Adapter<Object>() {
            public boolean eval(int n) {
                return n > 0;
            }
        }).length);
        System.out.println("%Diff="+(double)ArrayUtils.sum(ArrayUtils.and(f1, f2))/f1.length);

      */  
        
    }
}
