package com.syx.dlib;

import android.graphics.Bitmap;
import androidx.annotation.Keep;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import android.util.Log;

import java.util.Arrays;
import java.util.List;


public class FaceRecognize {
    private static final String TAG = "syx-dlib";

    // accessed by native methods
    static {
        try {
            System.loadLibrary("dlib-lib");
            jniNativeClassInit();
            Log.d(TAG, "jniNativeClassInit success");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "library not found");
        }
    }

    @SuppressWarnings("unused")
    public FaceRecognize() {
        jniInit();
    }

    @Nullable
    @WorkerThread
    public String detect(@NonNull Bitmap bitmap) {
        float[] facedes = new float[128];
        jniBitmapDetect(bitmap, facedes);
        return Arrays.toString(facedes);
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        release();
    }

    public void release() {
        jniDeInit();
    }

    @Keep
    private native static void jniNativeClassInit();

    @Keep
    private synchronized native int jniInit();

    @Keep
    private synchronized native int jniDeInit();

    @Keep
    private synchronized native void jniBitmapDetect(Bitmap bitmap, float[] facedescriptor);

}
