import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;



public class JavaOnlineModel {
    public static void main(String args[]){
        byte[] graphDef = loadTensorflowModel("./0631demo_rf.pb");

        System.out.println("数据: ");

        float inputs[][] = new float[10][1];
        for(int i = 0; i< 10; i++){
            for(int j =0; j< 1;j++){
                inputs[i][j] = i-5;
                System.out.println(inputs[i][j]);

            }
        }

        System.out.println("————————————————————————————————————————————");
        System.out.println("结果: ");

        Tensor<Float> input = covertArrayToTensor(inputs);
        Graph g = new Graph();
        g.importGraphDef(graphDef);
        Session s = new Session(g);
        Tensor result = s.runner().feed("inputx", input).fetch("output").run().get(0);

        long[] rshape = result.shape();
        int rs = (int) rshape[0];
        long realResult[] = new long[rs];
        result.copyTo(realResult);

        for(long a: realResult ) {
            System.out.println(a);
        }
    }
    static private byte[] loadTensorflowModel(String path){
        try {
            return Files.readAllBytes(Paths.get(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    static private Tensor<Float> covertArrayToTensor(float inputs[][]){
        return Tensors.create(inputs);
    }
}