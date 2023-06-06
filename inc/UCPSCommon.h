#ifndef	UCPSCOMMON_HEADER
#define UCPSCOMMON_HEADER

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#define CV_VERSION_ID	CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#if defined(_DEBUG)||defined(DEBUG)
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>  
#include <bitset>
#include <limits>
#include <ctime>
#include <exception>
#include <cmath>
#include <memory>

#include <stdio.h>
#include <stdlib.h>
#include <atlstr.h>
#include <stdarg.h> 
#include <conio.h>
#include <signal.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/video/video.hpp>  
#include <opencv2/videoio/videoio.hpp>  
#include <opencv2/videoio/legacy/constants_c.h>  
#include <opencv2/imgcodecs/legacy/constants_c.h>  

#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>

#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/mls.h>

#include <pcl/console/time.h>
#include <pcl/console/parse.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <boost/thread/thread.hpp>

#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkPLYWriter.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkImageViewer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkLight.h>
#include <vtkLightCollection.h>
#include <vtkRenderer.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkTriangle.h>

#pragma comment(lib, cvLIB("world"))

#endif
