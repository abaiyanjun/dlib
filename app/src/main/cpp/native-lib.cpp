#include <jni.h>
#include <string>

#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <dlib/rand.h>


#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
//#include <dlib/java/java_array.h>


#include<android/log.h>
#define LOG    "syx-jni" // 这个是自定义的LOG的标识
#define LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG,__VA_ARGS__) // 定义LOGD类型
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG,__VA_ARGS__) // 定义LOGI类型
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG,__VA_ARGS__) // 定义LOGW类型
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG,__VA_ARGS__) // 定义LOGE类型
#define LOGF(...)  __android_log_print(ANDROID_LOG_FATAL,LOG,__VA_ARGS__) // 定义LOGF类型

#include <android/bitmap.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>


#define SYX_FACE_JNI_METHOD(METHOD_NAME) \
  Java_com_syx_face_MainActivity_##METHOD_NAME

using namespace dlib;
using namespace std;
//using namespace cv;

#define MAKE_RGBA(r, g, b, a) (((a) << 24) | ((r) << 16) | ((g) << 8) | (b))
#define RGBA_A(p) (((p) & 0xFF000000) >> 24)
cv::CascadeClassifier ccf;


extern "C" JNIEXPORT jstring JNICALL
SYX_FACE_JNI_METHOD(stringFromJNI)(
        JNIEnv* env,
        jobject host /* this */) {


    //test dlib
    ostringstream seed;
    seed << (unsigned int)time(0);
    dlib::rand r;

    r.set_seed(seed.str());
    r.clear();

    unsigned int rint = r.get_random_32bit_number();

    std::string hello = "Hello from C++" + std::to_string(rint);

    LOGI(">>> %s", hello.data());

    return env->NewStringUTF(hello.c_str());
}



extern "C"
void JNICALL
SYX_FACE_JNI_METHOD(findface)(JNIEnv *env, jobject host, jobject bitmap) {
    if (bitmap == NULL) {
        return;
    }else{
        AndroidBitmapInfo info;
        memset(&info, 0, sizeof(info));
        AndroidBitmap_getInfo(env, bitmap, &info);
        void *pixels = NULL;
        AndroidBitmap_lockPixels(env, bitmap, &pixels);
        cv::Mat src(info.height, info.width, CV_8UC4, pixels);
        cvtColor(src, src, cv::COLOR_BGRA2RGB);

        cv::Mat img_v, img_hsv, img_struct,anchor_color_hsv, anchor_color, anchor1;
        int r = 30;
        double eps = 0.01;
        std::vector<cv::Mat> hsv_vec;
        cvtColor(src, img_hsv, cv::COLOR_BGR2HSV);


        dlib::shape_predictor pose_model;
        ccf.load("/sdcard/facelandmark/haarcascade_frontalface_alt.xml");
        dlib::deserialize("/sdcard/facelandmark/shape_predictor_68_face_landmarks.dat") >> pose_model;
        std::vector<cv::Rect> faces;
        cv::Mat gray;
        cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        ccf.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(300, 300), cv::Size(2000, 2000));
        for (std::vector<cv::Rect>::iterator iter = faces.begin(); iter != faces.end(); iter++)
            {
            cv::rectangle(src, *iter, cv::Scalar(0, 0, 255), 2, 8); //画出脸部矩形
        }
        dlib::cv_image<dlib::bgr_pixel> cimg(src);
        dlib::rectangle faces1;
        faces1.set_top(faces[0].tl().y);
        faces1.set_bottom(faces[0].br().y);
        faces1.set_left(faces[0].tl().x);
        faces1.set_right(faces[0].br().x);
        std::vector<dlib::full_object_detection> shapes;
        shapes.push_back(pose_model(cimg, faces1));

        if (!shapes.empty())
        {
            for (int j = 0; j < shapes.size(); j++)
            {
                for (int i = 0; i < 68; i++)
                {
                    circle(src, cv::Point(shapes[j].part(i).x(), shapes[j].part(i).y()),
                           10, cv::Scalar(0, 255, 0), -1);
                }
            }
        }

        cv::Mat output = src.clone();

        int a = 0, r1 = 0, g = 0, b = 0;
        for (int y = 0; y < info.height; ++y) {
            // From left to right
            for (int x = 0; x < info.width; ++x) {
                int *pixel = NULL;
                pixel = ((int *) pixels) + y * info.width + x;
                r1 = output.at<cv::Vec3b>(y, x)[0];
                g = output.at<cv::Vec3b>(y, x)[1];
                b = output.at<cv::Vec3b>(y, x)[2];
                a = RGBA_A(*pixel);
                *pixel = MAKE_RGBA(r1, g, b, a);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
    }
}


// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
        alevel0<
                alevel1<
                        alevel2<
                                alevel3<
                                        alevel4<
                                                max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
                                                        input_rgb_image_sized<150>
                                                >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------
extern "C"
void JNICALL
SYX_FACE_JNI_METHOD(findfaceVector)(JNIEnv *env, jobject host, jobject bitmap, jfloatArray javaArr) {
    if (bitmap == NULL) {
        return;
    }
    try{
        AndroidBitmapInfo info;
        memset(&info, 0, sizeof(info));
        AndroidBitmap_getInfo(env, bitmap, &info);
        void *pixels = NULL;
        AndroidBitmap_lockPixels(env, bitmap, &pixels);
        cv::Mat src(info.height, info.width, CV_8UC4, pixels);
        cv::cvtColor(src, src, cv::COLOR_BGRA2RGB);

        matrix<rgb_pixel> img;
        array2d<bgr_pixel> arrimg(src.rows, src.cols);
        dlib::assign_image(img, cv_image<rgb_pixel>(src));

        // The first thing we are going to do is load all our models.  First, since we need to
        // find faces in the image we will need a face detector:
        frontal_face_detector detector = get_frontal_face_detector();
        // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
        shape_predictor sp;
        deserialize("/sdcard/facelandmark/shape_predictor_5_face_landmarks.dat") >> sp;
        // And finally we load the DNN responsible for face recognition.
        anet_type net;
        deserialize("/sdcard/facelandmark/dlib_face_recognition_resnet_model_v1.dat") >> net;

        // test 10 times

        int lenght;
        float* arrp;

        for(int kk=0;kk<10;kk++)
        {
            LOGE(" test %d :  ", kk);

            // Run the face detector on the image of our action heroes, and for each face extract a
            // copy that has been normalized to 150x150 pixels in size and appropriately rotated
            // and centered.
            std::vector<matrix<rgb_pixel>> faces;
            for (auto face : detector(img))
            {
                auto shape = sp(img, face);
                matrix<rgb_pixel> face_chip;
                extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                faces.push_back(move(face_chip));
            }

            if (faces.size() == 0)
            {
                LOGE("%s", "No faces found in image!" );
                return ;
            }

            // This call asks the DNN to convert each face image in faces into a 128D vector.
            // In this 128D vector space, images from the same person will be close to each other
            // but vectors from different people will be far apart.  So we can use these vectors to
            // identify if a pair of images are from the same person or from different people.
            std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);


            // In particular, one simple thing we can do is face clustering.  This next bit of code
            // creates a graph of connected faces and then uses the Chinese whispers graph clustering
            // algorithm to identify how many people there are and which faces belong to whom.
            // dlib::serialize("/sdcard/facelandmark/101ramdump_mat1.dat") << ramdump(face_descriptors[0]);

            //获取Java数组长度
            lenght = env->GetArrayLength(javaArr);

            //根据Java数组创建C数组，也就是把Java数组转换成C数组
            arrp = env->GetFloatArrayElements(javaArr,0);


            LOGE("%s", "face_descriptors...");
            matrix<float, 0, 1> a = face_descriptors[0];
            int i=0;
            for (long r = 0; r < a.nr(); ++r)
            {
                for (long c = 0; c < a.nc(); ++c)
                {
                    LOGE(" %f ", a(r,c));
                    *(arrp+i++) = a(r,c);
                }
            }
        } //test 10 times kk

        //将C数组种的元素拷贝到Java数组中
        env->SetFloatArrayRegion(javaArr,0,lenght,arrp);

    }
    catch (std::exception& e)
    {
        LOGE("%s", e.what());
    }
}
// ----------------------------------------------------------------------------------------

//根据cv::Mat::type()的类型，使用不同的type_pixel加载到dlib::cv_image，
//并转换成dlib::matrix<dlib::bgr_pixel>格式
int cvMat2dlib(dlib::matrix<dlib::bgr_pixel> &dest, const cv::Mat &src)
{
    cv::Mat tmp;
    switch (src.type()) {
        case CV_8UC3:
            dlib::assign_image(dest, dlib::cv_image<dlib::bgr_pixel>(src));
            break;
        case CV_8UC4:
            tmp = src.clone();
            cv::cvtColor(tmp, tmp, cv::COLOR_BGRA2RGBA);
            dlib::assign_image(dest, dlib::cv_image<dlib::rgb_alpha_pixel>(tmp));
            break;
        case CV_8UC1:
            dlib::assign_image(dest, dlib::cv_image<unsigned char>(src));
            break;
        default:
            return -1;
            break;
    }

    return 0;
}


// ----------------------------------------------------------------------------------------
extern "C"
void JNICALL
SYX_FACE_JNI_METHOD(findfaceVectorBeta)(JNIEnv *env, jobject host, jobject bitmap, jfloatArray javaArr) {
    if (bitmap == NULL) {
        return;
    }
    try{
        int lenght;
        float* arrp;


        // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
        shape_predictor sp;
        deserialize("/sdcard/facelandmark/shape_predictor_5_face_landmarks.dat") >> sp;
        // And finally we load the DNN responsible for face recognition.
        anet_type net;
        deserialize("/sdcard/facelandmark/dlib_face_recognition_resnet_model_v1.dat") >> net;
        ccf.load("/sdcard/facelandmark/haarcascade_frontalface_alt.xml");

        LOGE("%s", "111");

        AndroidBitmapInfo info;
        memset(&info, 0, sizeof(info));
        AndroidBitmap_getInfo(env, bitmap, &info);
        void *pixels = NULL;
        AndroidBitmap_lockPixels(env, bitmap, &pixels);
        cv::Mat src(info.height, info.width, CV_8UC4, pixels);
        cvtColor(src, src, cv::COLOR_BGRA2RGB);

        cv::Mat img_v, img_hsv, img_struct,anchor_color_hsv, anchor_color, anchor1;
        int r = 30;
        double eps = 0.01;
        std::vector<cv::Mat> hsv_vec;
        cvtColor(src, img_hsv, cv::COLOR_BGR2HSV);

        LOGE("%s", "222");

        std::vector<cv::Rect> faces;
        cv::Mat gray;
        cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        ccf.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(300, 300), cv::Size(2000, 2000));

        LOGE("%s", "333");

        cv::Mat image_roi = src(faces[0]);
        //cv::cvtColor(image_roi, image_roi, cv::COLOR_BGRA2RGB);
        cv::Size dsize = cv::Size(150, 150);
        cv::Mat image_roi2 = cv::Mat(dsize, CV_8UC4);
        cv::resize(image_roi, image_roi2, dsize);

        LOGE("%s", "333 aaa");
        matrix<rgb_pixel> img;
        LOGE("%s", "333 bbb");
        dlib::assign_image(img, cv_image<rgb_pixel>(image_roi2));

        LOGE("%s", "444");

        std::vector<matrix<rgb_pixel>> dlib_faces;
        dlib_faces.push_back(img);
        std::vector<matrix<float, 0, 1>> face_descriptors;
        for(int n=0;n<4;n++) {
            LOGE("%s %d", "555 begin ", n);
            face_descriptors = net(dlib_faces);
            LOGE("%s %d", "555 end ", n);
        }
        //获取Java数组长度
        lenght = env->GetArrayLength(javaArr);

        //根据Java数组创建C数组，也就是把Java数组转换成C数组
        arrp = env->GetFloatArrayElements(javaArr,0);

        LOGE("%s", "face_descriptors...");
        matrix<float, 0, 1> a = face_descriptors[0];
        int i=0;
        for (long r = 0; r < a.nr(); ++r)
        {
            for (long c = 0; c < a.nc(); ++c)
            {
                LOGE(" %f ", a(r,c));
                *(arrp+i++) = a(r,c);
            }
        }

        //将C数组种的元素拷贝到Java数组中
        env->SetFloatArrayRegion(javaArr,0,lenght,arrp);
        LOGE("%s", "9999999");
    }
    catch (std::exception& e)
    {
        LOGE("%s", e.what());
    }
}
// ----------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------
// for com.syx.dlib.FaceRecognize native
#ifdef __cplusplus
extern "C" {
#endif

#define DLIB_FACE_JNI_METHOD(METHOD_NAME) \
  Java_com_syx_dlib_FaceRecognize_##METHOD_NAME

shape_predictor sp;
anet_type net;

void JNIEXPORT
DLIB_FACE_JNI_METHOD(jniNativeClassInit)(JNIEnv* env, jclass _this) {}


void JNICALL
DLIB_FACE_JNI_METHOD(jniBitmapDetect)(JNIEnv *env, jobject host,
        jobject bitmap, jfloatArray javaArr) {
    LOGE("jniBitmapFaceDet begin");

    if (bitmap == NULL) {
        return;
    }
    try{
        int lenght;
        float* arrp;

        LOGE("%s", "111");

        AndroidBitmapInfo info;
        memset(&info, 0, sizeof(info));
        AndroidBitmap_getInfo(env, bitmap, &info);
        void *pixels = NULL;
        AndroidBitmap_lockPixels(env, bitmap, &pixels);
        cv::Mat src(info.height, info.width, CV_8UC4, pixels);
        cvtColor(src, src, cv::COLOR_BGRA2RGB);

        cv::Mat img_v, img_hsv, img_struct,anchor_color_hsv, anchor_color, anchor1;
        int r = 30;
        double eps = 0.01;
        std::vector<cv::Mat> hsv_vec;
        cvtColor(src, img_hsv, cv::COLOR_BGR2HSV);

        LOGE("%s", "222");

        std::vector<cv::Rect> faces;
        cv::Mat gray;
        cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        ccf.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(300, 300), cv::Size(2000, 2000));

        if(faces.size()==0){
            return;
        }
        
        LOGE("%s %d", "333 faces.size()=", faces.size());

        cv::Mat image_roi = src(faces[0]);
        //cv::cvtColor(image_roi, image_roi, cv::COLOR_BGRA2RGB);
        cv::Size dsize = cv::Size(150, 150);
        cv::Mat image_roi2 = cv::Mat(dsize, CV_8UC4);
        cv::resize(image_roi, image_roi2, dsize);

        LOGE("%s", "333 aaa");
        matrix<rgb_pixel> img;
        LOGE("%s", "333 bbb");
        dlib::assign_image(img, cv_image<rgb_pixel>(image_roi2));

        LOGE("%s", "444");

        std::vector<matrix<rgb_pixel>> dlib_faces;
        dlib_faces.push_back(img);
        std::vector<matrix<float, 0, 1>> face_descriptors;

        LOGE("%s ", "555 begin ");
        face_descriptors = net(dlib_faces);
        LOGE("%s ", "555 end ");
        if(face_descriptors.size()==0){
            LOGE("%s ", "555 face_descriptors.size()==0 ");
            return;
        }

        LOGE("%s ", "666 begin ");
        double len = length(face_descriptors[0] - face_descriptors[0]*0.9);
        LOGE("%s %f", "666 end", len);

        //获取Java数组长度
        lenght = env->GetArrayLength(javaArr);

        //根据Java数组创建C数组，也就是把Java数组转换成C数组
        arrp = env->GetFloatArrayElements(javaArr,0);

        LOGE("%s", "face_descriptors...");
        matrix<float, 0, 1> a = face_descriptors[0];
        int i=0;
        for (long r = 0; r < a.nr(); ++r)
        {
            for (long c = 0; c < a.nc(); ++c)
            {
                //LOGE(" %f ", a(r,c));
                *(arrp+i++) = a(r,c);
            }
        }

        //将C数组种的元素拷贝到Java数组中
        env->SetFloatArrayRegion(javaArr,0,lenght,arrp);
        LOGE("%s", "9999999");
    }
    catch (std::exception& e)
    {
        LOGE("%s", e.what());
    }

    LOGE("jniBitmapFaceDet end ");
    return;
}

jint JNIEXPORT JNICALL DLIB_FACE_JNI_METHOD(jniInit)(JNIEnv* env, jobject thiz) {
    LOGE("jniInit");
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
    deserialize("/sdcard/facelandmark/shape_predictor_5_face_landmarks.dat") >> sp;
    // And finally we load the DNN responsible for face recognition.
    deserialize("/sdcard/facelandmark/dlib_face_recognition_resnet_model_v1.dat") >> net;
    ccf.load("/sdcard/facelandmark/haarcascade_frontalface_alt.xml");

    return JNI_OK;
}

jint JNIEXPORT JNICALL
DLIB_FACE_JNI_METHOD(jniDeInit)(JNIEnv* env, jobject thiz) {
    LOGE("jniDeInit");
    return JNI_OK;
}

#ifdef __cplusplus
}
#endif
// ----------------------------------------------------------------------------------------



