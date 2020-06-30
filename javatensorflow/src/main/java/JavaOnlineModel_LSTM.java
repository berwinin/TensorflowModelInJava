import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Created by 刘建平pinard on 2018/7/1.
 */
public class JavaOnlineModel_LSTM {
    public static void main(String args[]){
        byte[] graphDef = loadTensorflowModel("/Users/liu_bowen/Desktop/Trusfort/git/Trusfort/lbw/javatensorflow/src/main/java/offlineLSTM.pb");


        float inputs[][] = {{0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0.00784314f,0.55686277f,0.98039222f,0.20392159f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0.08627451f,0.59607846f,0.99607849f,0.99607849f,0.33333334f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0.f,0.09803922f,0.55686277f,0.99607849f,0.99607849f,0.75686282f,0.11764707f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0.01568628f,0.57647061f,0.99607849f,0.99607849f,0.57647061f,0.02352941f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0.5411765f, 0.99607849f,0.99607849f,0.59215689f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0.50196081f,0.99215692f,0.99607849f,0.57254905f,0.01960784f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0.f,0.21960786f,0.92549026f,0.99607849f,0.59215689f,0.00784314f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0.72549021f,0.99607849f,0.69411767f,0.10588236f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0.90196085f,0.99607849f,0.3019608f, 0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0.f,0.29411766f,0.99607849f,0.67058825f,0.00392157f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0.61176473f,0.99607849f,0.34901962f,0f,0f,0f,0.f,0f,0f,0f,0.00784314f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0.68627453f,0.99607849f,0.27450982f,0f,0f,0f,0.f,0f,0.22352943f,0.627451f,0.73333335f,0.07058824f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0.f,0.06666667f,0.94901967f,0.99607849f,0.27450982f,0f,0f,0.f,0.04313726f,0.84313732f,0.97254908f,0.99607849f,0.99607849f,0.62352943f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0.08235294f,0.99607849f,0.99607849f,0.27450982f,0f,0.f,0.34901962f,0.80392164f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.68235296f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0.04313726f,0.84313732f,0.99607849f,0.27450982f,0.05490196f,0.39607847f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.64313728f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0.68627453f,0.99607849f,0.7843138f, 0.86666673f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.80392164f,0.08627451f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0.28627452f,0.98431379f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.98823535f,0.59607846f,0.08627451f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0.90196085f,0.99607849f,0.99607849f,0.99607849f,0.99607849f,0.99215692f,0.64313728f,0.29803923f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0.90196085f,0.99607849f,0.61176473f,0.43137258f,0.04313726f,0.03529412f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0.90196085f,0.42745101f,0.01176471f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0f,0f,0f,0f,0f,0.f,0f,0.f}};
//        float inputs[][] = new float[2][784];
//        for(int j =0; j< 2;j++){
//            for(int k=0;k<784;k++){
//                float rd=(float)Math.random();
////                System.out.println("rd=== "+rd);
//                inputs[j][k] = rd;
//            }
//
//        }
//        System.out.println(inputs);

        Tensor<Float> input = covertArrayToTensor(inputs);
        Graph g = new Graph();
        g.importGraphDef(graphDef);
        Session s = new Session(g);

        Tensor result = s.runner().feed("inputxx", input).fetch("outputxx").run().get(0);

        long[] rshape = result.shape();
        System.out.println("~~~~~~~~~~"+Arrays.toString(rshape)); // 1行10列 [1, 10]

        int rs = (int) rshape[0];

        long realResult[] = new long[rs];
        result.copyTo(realResult);
        int ll = 1;
        for(long a: realResult ) {
            System.out.println("--------  "+ll+"  +++++++++++++");
            ll += 1;
            System.out.println(a);
        }



        // 自己添加
        Tensor result_pre = s.runner().feed("inputxx", input).fetch("output_pre").run().get(0);

        long[] rshape_pre = result_pre.shape();
        System.out.println("~~~~~~~~~~"+Arrays.toString(rshape_pre)); // 1行10列 [1, 10]

        int rs_pre_a = (int) rshape_pre[0];
        int rs_pre_b = (int) rshape_pre[1];
        System.out.println("**************"+rs_pre_a+"&&&&&&&&&&&"+rs_pre_b);
        float realResult_pre[][] = new float[rs_pre_a][rs_pre_b];
        result_pre.copyTo(realResult_pre);
        int qq = 1;
        for(float[] a_pre: realResult_pre ) {
            System.out.println("----pre===----  "+qq+"  +++++++++++++");
            qq += 1;
            System.out.println(Arrays.toString(a_pre));
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