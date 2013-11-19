#include <cv.h>
#include <cvaux.h>
#include <highgui.h>

#include <cstring>
#include <cmath>

#include <vector>

#include "flandmark_detector.h"
#include "json/json.h"

using namespace std;

/*
 */
void drawVectorFromFaceCenter( CvArr *orig,
                              double v[4] ,
                              const Json::Value &currentFace,
                              CvScalar color,
                              const cv::Matx34d &Rt,
                              const cv::Mat &cameraMatrix) {
    cv::Matx41d originOnMesh;
    originOnMesh(0, 0) = 0;
    originOnMesh(1, 0) = 0;
    originOnMesh(2, 0) = 0;
    originOnMesh(3, 0) = 1;
    cv::Matx31d originOnImage = Rt*originOnMesh;
    
    cv::Matx41d endpointOnMesh;
    endpointOnMesh(0, 0) = v[0];
    endpointOnMesh(1, 0) = v[1];
    endpointOnMesh(2, 0) = v[2];
    endpointOnMesh(3, 0) = v[3];
    cv::Matx31d endpointOnImage = Rt*endpointOnMesh;
    
    cv::Mat normalizedPointOnImage;
    
    cv::normalize(cv::Mat(endpointOnImage - originOnImage), normalizedPointOnImage);
    normalizedPointOnImage = normalizedPointOnImage * 100;
    
    int end_x = cv::Vec3d(normalizedPointOnImage)(0)
    + currentFace["face-center"]["x"].asDouble();
    int end_y = cv::Vec3d(normalizedPointOnImage)(1) +
    currentFace["face-center"]["y"].asDouble();
    
    cvLine(orig,
           cvPoint(currentFace["face-center"]["x"].asDouble(),
                   currentFace["face-center"]["y"].asDouble()),
           cvPoint(end_x, end_y), color);
}

cv::Vec3d applyTransform(double v[4],
                         const cv::Matx34d &Rt) {
    cv::Matx41d originOnMesh;
    originOnMesh(0, 0) = 0;
    originOnMesh(1, 0) = 0;
    originOnMesh(2, 0) = 0;
    originOnMesh(3, 0) = 1;
    cv::Matx31d originOnImage = Rt*originOnMesh;
    
    cv::Matx41d endpointOnMesh;
    endpointOnMesh(0, 0) = v[0];
    endpointOnMesh(1, 0) = v[1];
    endpointOnMesh(2, 0) = v[2];
    endpointOnMesh(3, 0) = v[3];
    cv::Matx31d endpointOnImage = Rt*endpointOnMesh;
    
    cv::Mat normalizedPointOnImage;
    
    cv::normalize(cv::Mat(endpointOnImage - originOnImage), normalizedPointOnImage);
    
    return cv::Vec3d(normalizedPointOnImage);
}

Json::Value detectFaceInImage(IplImage *orig,
                              IplImage* input,
                              CvHaarClassifierCascade* cascade,
                              FLANDMARK_Model *model,
                              int *bbox,
                              double *landmarks)
{
    Json::Value root(Json::arrayValue);
    
    // Smallest face size.
    CvSize minFeatureSize = cvSize(40, 40);
    int flags =  CV_HAAR_DO_CANNY_PRUNING;
    // How detailed should the search be.
    float search_scale_factor = 1.1f;
    CvMemStorage* storage;
    CvSeq* rects;
    int nFaces;
    
    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);
    
    // Detect all the faces in the greyscale image.
    rects = cvHaarDetectObjects(input,
                                cascade,
                                storage,
                                search_scale_factor,
                                2,
                                flags,
                                minFeatureSize);
    nFaces = rects->total;
    
    double t = (double)cvGetTickCount();
    
    for (int iface = 0; iface < (rects ? nFaces : 0); ++iface) {
        Json::Value currentFace;
        
        CvRect *r = (CvRect*)cvGetSeqElem(rects, iface);
        
        bbox[0] = r->x;
        bbox[1] = r->y;
        bbox[2] = r->x + r->width;
        bbox[3] = r->y + r->height;
        
        flandmark_detect(input,
                         bbox,
                         model,
                         landmarks);
        
        // display landmarks
        cvRectangle(orig, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]), CV_RGB(255,0,0) );
        
        Json::Value faceRect;
        
        faceRect["top-left-x"] = bbox[0];
        faceRect["top-left-y"] = bbox[1];
        faceRect["bottom-right-x"] = bbox[2];
        faceRect["bottom-right-y"] = bbox[3];
        
        currentFace["face-rect"] = faceRect;
        
        cvRectangle(orig, cvPoint(model->bb[0], model->bb[1]), cvPoint(model->bb[2], model->bb[3]), CV_RGB(0,0,255) );
        
        Json::Value flandmarkFaceRect;
        
        flandmarkFaceRect["top-left-x"] = model->bb[0];
        flandmarkFaceRect["top-left-y"] = model->bb[1];
        flandmarkFaceRect["bottom-right-x"] = model->bb[2];
        flandmarkFaceRect["bottom-right-y"] = model->bb[3];
        
        currentFace["flandmark-face-rect"] = flandmarkFaceRect;
        
        cvCircle(orig, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0, 0,255), CV_FILLED);
        
        Json::Value faceCenter;
        faceCenter["x"] = landmarks[0];
        faceCenter["y"] = landmarks[1];
        
        currentFace["face-center"] = faceCenter;
        
        Json::Value canthusRl;
        Json::Value canthusLr;
        Json::Value mouthCornerR;
        Json::Value mouthCornerL;
        Json::Value canthusRr;
        Json::Value canthusLl;
        Json::Value nose;
        
        for (int i = 2; i < 2*model->data.options.M; i += 2)
        {
            cvCircle(orig, cvPoint(int(landmarks[i]), int(landmarks[i+1])), 3, CV_RGB(255,0,0), CV_FILLED);
            
            switch(i/2)
            {
                case 1 :
                    canthusRl["x"] = landmarks[i];
                    canthusRl["y"] = landmarks[i+1];
                    currentFace["canthus-rl"] = canthusRl;
                    break;
                case 2 :
                    canthusLr["x"] = landmarks[i];
                    canthusLr["y"] = landmarks[i+1];
                    currentFace["canthus-lr"] = canthusLr;
                    break;
                case 3:
                    mouthCornerR["x"] = landmarks[i];
                    mouthCornerR["y"] = landmarks[i+1];
                    currentFace["mouth-corner-r"] = mouthCornerR;
                    break;
                case 4:
                    mouthCornerL["x"] = landmarks[i];
                    mouthCornerL["y"] = landmarks[i+1];
                    currentFace["mouth-corner-l"] = mouthCornerL;
                    break;
                case 5:
                    canthusRr["x"] = landmarks[i];
                    canthusRr["y"] = landmarks[i+1];
                    currentFace["canthus-rr"] = canthusRr;
                    break;
                case 6:
                    canthusLl["x"] = landmarks[i];
                    canthusLl["y"] = landmarks[i+1];
                    currentFace["canthus-ll"] = canthusLl;
                    break;
                case 7:
                    nose["x"] = landmarks[i];
                    nose["y"] = landmarks[i+1];
                    currentFace["nose"] = nose;
                    break;
            }
        }
        
        cv::Mat rvec, tvec;
        
        // landmarks coordiates from 3D model
        
        float modX[7]={-0.23260, 1.19237,  -0.60929, 1.53890,  -1.32542, 2.38696,  0.54571 };
        float modY[7]={-6.82082, -6.80873, -6.43984, -6.36571, -6.62015, -6.48895, -8.01948};
        float modZ[7]={71.57423, 71.47037, 68.69979, 68.66721, 71.55910, 71.48106, 69.76753};
        
        vector<cv::Point3f> model_points;
        
        for (int i=0;i<7;i++) {
            model_points.push_back(cv::Point3f(modX[i],modY[i],modZ[i]));
        }
        
        // detected landmark coordinates
        
        
        float imX[7]={ currentFace["canthus-rl"]["x"].asDouble(),
            currentFace["canthus-lr"]["x"].asDouble(),
            currentFace["mouth-corner-r"]["x"].asDouble(),
            currentFace["mouth-corner-l"]["x"].asDouble(),
            currentFace["canthus-rr"]["x"].asDouble(),
            currentFace["canthus-ll"]["x"].asDouble(),
            currentFace["nose"]["x"].asDouble()};
        float imY[7]={ currentFace["canthus-rl"]["y"].asDouble(),
            currentFace["canthus-lr"]["y"].asDouble(),
            currentFace["mouth-corner-r"]["y"].asDouble(),
            currentFace["mouth-corner-l"]["y"].asDouble(),
            currentFace["canthus-rr"]["y"].asDouble(),
            currentFace["canthus-ll"]["y"].asDouble(),
            currentFace["nose"]["y"].asDouble()};
        
        vector<cv::Point2f> image_points;
        
        for (int i=0;i<7;i++)
        {
            image_points.push_back(cv::Point2f(imX[i],imY[i]));
        }
        
        
        cv::Mat cameraMatrix = cv::Mat((CvMat*)cvLoad( "Intrinsics.xml" ));
        cv::Mat distCoeffs = cv::Mat((CvMat*)cvLoad( "Distortion.xml" ));
        
        
        solvePnPRansac(cv::Mat(model_points),
                       cv::Mat(image_points),
                       cameraMatrix, distCoeffs, rvec, tvec, false);
        
        cv::Matx33d R;
        cv::Rodrigues(rvec, R);
        
        cv::Matx34d Rt;
        
        for(int i=0; i<3; i++)
        {
            for(int j=0; j<4; j++)
            {
                if(j < 3)
                {
                    Rt(i, j) = R(i, j);
                } else
                {
                    Rt(i, j) = cv::Vec3d(tvec)(i);
                }
            }
        }
        
        double minusYVector[4] = {0, -1, 0, 1};
        double xVector[4] = {1, 0, 0, 1};
        double yVector[4] = {0, 1, 0, 1};
        double zVector[4] = {0, 0, 1, 1};
        
        if(applyTransform(minusYVector, Rt)(1) >= 0) {
            drawVectorFromFaceCenter(orig, minusYVector, currentFace, CV_RGB(255, 255, 255), Rt, cameraMatrix);
            drawVectorFromFaceCenter(orig, xVector, currentFace, CV_RGB(255, 0, 0), Rt, cameraMatrix);
            drawVectorFromFaceCenter(orig, yVector, currentFace, CV_RGB(0, 255, 0), Rt, cameraMatrix);
            drawVectorFromFaceCenter(orig, zVector, currentFace, CV_RGB(0, 0, 255), Rt, cameraMatrix);
        } else {
            cv::Matx44d reverseY = cv::Matx44d::zeros();
            reverseY(0,0) = 1;
            reverseY(1,1) = -1;
            reverseY(2,2) = 1;
            reverseY(3,3) = 1;
            Rt = Rt*reverseY;
            drawVectorFromFaceCenter(orig, minusYVector, currentFace, CV_RGB(255, 255, 255), Rt, cameraMatrix);
            drawVectorFromFaceCenter(orig, xVector, currentFace, CV_RGB(255, 0, 0), Rt, cameraMatrix);
            drawVectorFromFaceCenter(orig, yVector, currentFace, CV_RGB(0, 255, 0), Rt, cameraMatrix);
            drawVectorFromFaceCenter(orig, zVector, currentFace, CV_RGB(0, 0, 255), Rt, cameraMatrix);
        }
        
        cv::Mat tmp,tmp1,tmp2,tmp3,tmp4,tmp5;
        cv::Vec3d eav;
        cv::decomposeProjectionMatrix(Rt, tmp, tmp1, tmp2, tmp3, tmp4, tmp5, eav);
        
        currentFace["roll"] = eav[0];
        currentFace["pitch"] = eav[1];
        currentFace["yaw"] = eav[3];
        
        root.append(currentFace);
    }
    t = (double)cvGetTickCount() - t;
    int ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
    
    if (nFaces > 0)
    {
        printf("Faces detected: %d; Detection of facial landmark on all faces took %d ms\n", nFaces, ms);
    } else {
        printf("NO Face\n");
    }
    
    cvReleaseMemStorage(&storage);
    
    return root;
}

int main( int argc, char** argv )
{
    char flandmark_window[] = "flandmark_example1";
    double t;
    int ms;
    
    if (argc < 1)
    {
        fprintf(stderr, "Usage: flandmark_1 <path_to_input_image> \n");
        exit(1);
    }
    
    cvNamedWindow(flandmark_window, 0);
    
    // Haar Cascade file, used for Face Detection.
    char faceCascadeFilename[] = "haarcascade_frontalface_alt.xml";
    // Load the HaarCascade classifier for face detection.
    CvHaarClassifierCascade* faceCascade;
    faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0);
    if( !faceCascade )
    {
        printf("Couldnt load Face detector '%s'\n", faceCascadeFilename);
        exit(1);
    }
    
    // ------------- begin flandmark load model
    t = (double)cvGetTickCount();
    FLANDMARK_Model * model = flandmark_init("flandmark_model.dat");
    
    if (model == 0)
    {
        printf("Structure model wasn't created. Corrupted file flandmark_model.dat?\n");
        exit(1);
    }
    
    t = (double)cvGetTickCount() - t;
    ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
    printf("Structure model loaded in %d ms.\n", ms);
    // ------------- end flandmark load model
    
    // input image
    IplImage *frame = cvLoadImage(argv[1]);
    if (frame == NULL)
    {
        fprintf(stderr, "Cannot open image %s. Exiting...\n", argv[1]);
        exit(1);
    }
    // convert image to grayscale
    IplImage *frame_bw = cvCreateImage(cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1);
    cvConvertImage(frame, frame_bw);
    
    int *bbox = (int*)malloc(4*sizeof(int));
    double *landmarks = (double*)malloc(2*model->data.options.M*sizeof(double));
    Json::Value root = detectFaceInImage(frame, frame_bw, faceCascade, model, bbox, landmarks);
    
    cvShowImage(flandmark_window, frame);
    cvWaitKey(0);
    
    if (argc == 3)
    {
        printf("Saving image to file %s...\n", argv[2]);
        cvSaveImage(argv[2], frame);
    }
    
    
    Json::StyledWriter writer;
    std::cout << root;
    
    // cleanup
    free(bbox);
    free(landmarks);
    cvDestroyWindow(flandmark_window);
    cvReleaseImage(&frame);
    cvReleaseImage(&frame_bw);
    cvReleaseHaarClassifierCascade(&faceCascade);
    flandmark_free(model);
}
