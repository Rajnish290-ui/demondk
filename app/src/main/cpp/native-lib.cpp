#include <jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <android/log.h>
#include <cstdio>

constexpr int ITUR_BT_601_CY = 1220542;
constexpr int ITUR_BT_601_CUB = 2116026;
constexpr int ITUR_BT_601_CUG = -409993;
constexpr int ITUR_BT_601_CVG = -852492;
constexpr int ITUR_BT_601_CVR = 1673527;
constexpr int ITUR_BT_601_SHIFT = 20;


struct PlaneInfo {
    const uint8_t* ptr;
    int pixelStride;
    int rowStride;
};

enum class YUV_FORMAT { UNSUPPORTED, YUV_420_888 };

struct YUVInfo {
    YUV_FORMAT format = YUV_FORMAT::UNSUPPORTED;
    int width = 0;
    int height = 0;
    std::array<PlaneInfo, 3> planes;
};

struct  YUV420p2RGBA : cv::ParallelLoopBody {
    const uint8_t* _my;
    const uint8_t* _mu;
    const uint8_t* _mv;
    const int _yRowStride;
    const int _uRowStride;
    const int _vRowStride;
    const int _yPixelStride;
    const int _uPixelStride;
    const int _vPixelStride;
    cv::Mat& _output;

    YUV420p2RGBA(cv::Mat& output, const YUVInfo& yuvInfo)
            : _output(output),
              _my(yuvInfo.planes[0].ptr),
              _mu(yuvInfo.planes[1].ptr),
              _mv(yuvInfo.planes[2].ptr),
              _yRowStride(yuvInfo.planes[0].rowStride),
              _uRowStride(yuvInfo.planes[1].rowStride),
              _vRowStride(yuvInfo.planes[2].rowStride),
              _yPixelStride(yuvInfo.planes[0].pixelStride),
              _uPixelStride(yuvInfo.planes[1].pixelStride),
              _vPixelStride(yuvInfo.planes[2].pixelStride) {
    }

    void operator()(const cv::Range& range) const override {
        const auto width = _output.cols;
        for (int j = range.start; j < range.end; j++) {
            auto row1 = _output.ptr<uint8_t>(2 * j);
            auto row2 = _output.ptr<uint8_t>(2 * j + 1);

            auto y1 = _my + 2 * j * _yRowStride;
            auto y2 = y1 + _yRowStride;

            auto u1 = _mu + j * _uRowStride;
            auto v1 = _mv + j * _vRowStride;

            for (int i = 0; i < width; i += 2) {
                const auto u = static_cast<int>(*u1) - 128;
                const auto v = static_cast<int>(*v1) - 128;

                const auto ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                const auto guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u;
                const auto buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                const auto y00 = std::max(0, static_cast<int>(*y1) - 16) * ITUR_BT_601_CY;
                row1[0] = cv::saturate_cast<uint8_t>((y00 + buv) >> ITUR_BT_601_SHIFT);
                row1[1] = cv::saturate_cast<uint8_t>((y00 + guv) >> ITUR_BT_601_SHIFT);
                row1[2] = cv::saturate_cast<uint8_t>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                row1[3] = uint8_t{0xff};

                y1 += _yPixelStride;

                const auto y01 = std::max(0, static_cast<int>(*y1) - 16) * ITUR_BT_601_CY;
                row1[4] = cv::saturate_cast<uint8_t>((y01 + buv) >> ITUR_BT_601_SHIFT);
                row1[5] = cv::saturate_cast<uint8_t>((y01 + guv) >> ITUR_BT_601_SHIFT);
                row1[6] = cv::saturate_cast<uint8_t>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                row1[7] = uint8_t{0xff};

                y1 += _yPixelStride;

                const auto y10 = std::max(0, static_cast<int>(*y2) - 16) * ITUR_BT_601_CY;
                row2[0] = cv::saturate_cast<uint8_t>((y10 + buv) >> ITUR_BT_601_SHIFT);
                row2[1] = cv::saturate_cast<uint8_t>((y10 + guv) >> ITUR_BT_601_SHIFT);
                row2[2] = cv::saturate_cast<uint8_t>((y10 + ruv) >> ITUR_BT_601_SHIFT);
                row2[3] = uint8_t{0xff};

                y2 += _yPixelStride;

                const auto y11 = std::max(0, static_cast<int>(*y2) - 16) * ITUR_BT_601_CY;
                row2[4] = cv::saturate_cast<uint8_t>((y11 + buv) >> ITUR_BT_601_SHIFT);
                row2[5] = cv::saturate_cast<uint8_t>((y11 + guv) >> ITUR_BT_601_SHIFT);
                row2[6] = cv::saturate_cast<uint8_t>((y11 + ruv) >> ITUR_BT_601_SHIFT);
                row2[7] = uint8_t{0xff};

                y2 += _yPixelStride;

                row1 += 8;
                row2 += 8;
                u1 += _uPixelStride;
                v1 += _vPixelStride;
            }
        }
    }
};


inline jobject callVoidObjectMethod(JNIEnv *env, jobject obj, const char* name, const char* signature) {
    assert(env != nullptr);
    jclass cl = env->GetObjectClass(obj);
    jmethodID methodID = env->GetMethodID(cl, name, signature);
    return env->CallObjectMethod(obj, methodID);
}

inline int callVoidIntMethod(JNIEnv *env, jobject obj, const char* name) {
    assert(env != nullptr);
    jclass cl = env->GetObjectClass(obj);
    jmethodID methodID = env->GetMethodID(cl, name, "()I");
    return static_cast<int>( env->CallIntMethod(obj, methodID) );
}

inline void* getBufferAddress(JNIEnv* env, jobject b) {
    assert(env != nullptr);
    return env->GetDirectBufferAddress(b);
}

cv::Mat YUV2BGRA(JNIEnv *env, jobject image){
    auto yuvInfo = YUVInfo{};

    int width = 0;
    int height = 0;

    width  = callVoidIntMethod(env, image, "getWidth");
    height = callVoidIntMethod(env, image, "getHeight");
    bool isYUV = callVoidIntMethod(env, image, "getFormat")==35;

    auto planesArray = (jobjectArray)callVoidObjectMethod(env, image, "getPlanes", "()[Landroid/media/Image$Plane;");

    for(int i=0; i<3; ++i) {
        auto plane = env->GetObjectArrayElement(planesArray, i);
        auto buffer = callVoidObjectMethod(env, plane, "getBuffer", "()Ljava/nio/ByteBuffer;");

        yuvInfo.planes[i].pixelStride = callVoidIntMethod(env, plane, "getPixelStride");
        yuvInfo.planes[i].rowStride   = callVoidIntMethod(env, plane, "getRowStride");
        yuvInfo.planes[i].ptr         = static_cast<uint8_t *>(getBufferAddress(env, buffer));
    }

    if (width <= 0 || height <= 0) {
        return {};
    }

    if (!isYUV) {
        return {};
    }

    auto output = cv::Mat(height, width, CV_8UC4);

    cv::parallel_for_(cv::Range{0, output.rows / 2},YUV420p2RGBA{output, yuvInfo});

    return output;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_demondk_MainActivity_imageFromJNI(JNIEnv *env, jobject thiz, jobject image) {
    // TODO: implement imageFromJNI()
    __android_log_print(ANDROID_LOG_INFO, "JNI", "Start");
    auto img = YUV2BGRA(env, image);
    auto msg ="Image "+std::to_string(img.cols)+" "+std::to_string(img.rows);
    __android_log_print(ANDROID_LOG_INFO, "JNI Done", "%s", msg.c_str());

}