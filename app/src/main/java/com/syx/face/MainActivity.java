package com.syx.face;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Keep;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;

import com.syx.dlib.Constants;
import com.syx.dlib.FaceDet;
import com.syx.dlib.PedestrianDet;
import com.syx.dlib.VisionDetRet;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import android.os.Bundle;
import android.view.View;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import android.graphics.BitmapFactory;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.widget.ImageView;
import android.widget.Button;
import android.content.Intent;
import android.net.Uri;

import java.io.FileNotFoundException;


public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private static final String TAG = "syx-dlib";
    FaceDet mFaceDet;
    PedestrianDet mPersonDet;
    private Button button ;
    private Button button1 ;
    private ImageView picture;
    private Uri imageUri;
    private TextView tv;
    public static File tempFile;


    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    private void copyAssetsFile(String name, File dir) throws IOException {
        if (!dir.exists()) {
            dir.mkdirs();
        }
        File file = new File(dir, name);
        if (!file.exists()){
            InputStream is = getAssets().open(name);
            FileOutputStream fos = new FileOutputStream(file);
            int len;
            byte[] buffer = new byte[2048];
            while ((len = is.read(buffer)) != -1)
                fos.write(buffer, 0, len);
            fos.close();
            is.close();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        button = findViewById(R.id.button);
        button1 = findViewById(R.id.button1);
        picture = findViewById(R.id.image);
        button.setOnClickListener(this);
        button1.setOnClickListener(this);

        try {
            File dir = new File("/sdcard/facelandmark");
            copyAssetsFile("haarcascade_frontalface_alt.xml", dir);
            copyAssetsFile("shape_predictor_68_face_landmarks.dat", dir);
            copyAssetsFile("shape_predictor_5_face_landmarks.dat", dir);
            copyAssetsFile("dlib_face_recognition_resnet_model_v1.dat", dir);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Example of a call to a native method
        tv = findViewById(R.id.sample_text);
        tv.setText(stringFromJNI());
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
    public void onClick(View view){
        if(view.getId()==R.id.button){
            openGallery();
        }
        if(view.getId()==R.id.button1){
            ImageView tv = findViewById(R.id.image);
            tv.setImageBitmap(function1());
        }
    }

    public Bitmap  function1(){
        Bitmap bitmap =((BitmapDrawable)picture.getDrawable()).getBitmap();
        float[] facedes = new float[128];
        long t1=System.currentTimeMillis();
        //findfaceVector(bitmap, facedes);
        findfaceVectorBeta(bitmap, facedes);
        long t2=System.currentTimeMillis();

        tv.setText((t2-t1) + " \n " +Arrays.toString(facedes));

        findface(bitmap);

        return bitmap;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            try {
                if (data != null) {
                    Uri uri = data.getData();
                    imageUri = uri;
                }
                Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
                picture.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }

    }

    public void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, 1);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native void findface(Bitmap bitmap);


    @NonNull
    protected void runDetectAsync(@NonNull String imgPath) {

        final String targetPath = Constants.getFaceShapeModelPath();
        if (!new File(targetPath).exists()) {
            throw new RuntimeException("cannot find shape_predictor_68_face_landmarks.dat");
        }
        // Init
        if (mPersonDet == null) {
            mPersonDet = new PedestrianDet();
        }
        if (mFaceDet == null) {
            mFaceDet = new FaceDet(Constants.getFaceShapeModelPath());
        }

        Log.d(TAG, "Image path: " + imgPath);
        List<VisionDetRet> faceList = mFaceDet.detect(imgPath);
        if (faceList.size() > 0) {
            Log.d(TAG, "faceList.size(): " + faceList.size());
        } else {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(getApplicationContext(), "No face", Toast.LENGTH_SHORT).show();
                }
            });
        }

    }

    public native void findfaceVector(Bitmap bitmap, float[] face_descriptor);

    public native void findfaceVectorBeta(Bitmap bitmap, float[] face_descriptor);

}
