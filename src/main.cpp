

#include "UCPSCommon.h"
#include "UCPSCLS.h"

#define EXPORT_OUTPUT_PLY_ON
#define EXPORT_OUTPUT_OBJ_ON

int main(int argc, char** argv)
{
	int64 elapse_t1 = cv::getTickCount();
	std::shared_ptr<clsUCPSD1> pUCPSD1 = std::make_shared<clsUCPSD1>();
	std::shared_ptr<clsUCPS> pclsUCPS = pUCPSD1;
	  /* create capture device (webcam on Macbook Pro) */
	std::vector<cv::Mat> camImages(IMAGE_MAX_NUMBER);

	try {
		  /* using asset images */
		for (int i = 0; i < IMAGE_MAX_NUMBER; i++) {
			cv::String filename = cv::format("../res/koreaCoin%d.bmp", i + 1);
			camImages[i] = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		}
		/* threshold images */
		//cv::Mat Mask = imageMask(camImages);
		cv::Mat Mask = cv::Mat::zeros(camImages[0].rows, camImages[0].cols, CV_8UC1);
		cv::circle(Mask, cv::Point(camImages[0].cols >> 1, camImages[0].rows >> 1), std::min(camImages[0].cols >> 1, camImages[0].rows >> 1), cv::Scalar(255, 0, 0), cv::FILLED);
		//  cv::imshow("Mask", Mask);
		/* compute normal map */
		cv::Mat S = pUCPSD1->computeNormals(camImages, Mask);
		cv::Mat Normalmap;
		cv::cvtColor(S, Normalmap, CV_BGR2RGB);
		cv::String filenameNormal = "../output/normalmap.png";
		cv::imwrite(filenameNormal, Normalmap);
		std::cout << filenameNormal<<" saved" << std::endl;
		/* compute depth map */
		cv::Mat Depth = pUCPSD1->localHeightfield(S);
		cv::Mat dst = cv::Mat(Depth.size(), CV_32FC1, cv::Scalar::all(255));
		Depth.copyTo(dst, Mask);
		cv::Mat DepthTemp = pclsUCPS->cvtFloatToGrayscale(dst);
		cv::String filenameDepth = "../output/LocalDepthmap.png";
		cv::imwrite(filenameDepth, DepthTemp);
		std::cout << filenameDepth <<" saved" << std::endl;
		cv::Mat DepthPS;
		DepthTemp.convertTo(DepthPS, CV_32FC1);
#ifdef EXPORT_OUTPUT_PLY_ON
		std::cout << "exporting PLY file ..." << std::endl;
		pclsUCPS->exportPLY(DepthPS.cols, DepthPS.rows, DepthPS);
#endif
#ifdef EXPORT_OUTPUT_OBJ_ON
		std::cout << "exporting OBJ file ..." << std::endl;
		pclsUCPS->exportOBJ(dst, S, camImages[0]);
#endif
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		// Fill in the cloud data
		cloud->width = DepthPS.cols;
		cloud->height = DepthPS.rows;
		cloud->resize(cloud->width * cloud->height);
		cloud->is_dense = true;
		int idx = 0;
		for (int i = 0; i < DepthPS.rows; i++)
			for (int j = 0; j < DepthPS.cols; j++)
			{
				cloud->points[idx].x = int(j + 1);
				cloud->points[idx].y = int(i + 1);
				cloud->points[idx].z = 255 - int(DepthPS.at<float>(i, j));
				idx++;
			}

		std::cout << "visualizing PCD viewer..." << std::endl;
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(cloud, "z");
		viewer->addPointCloud<pcl::PointXYZ>(cloud, fildColor, "sample cloud");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");

		//use the following functions to get access to the underlying more advanced/powerful
		//PCLVisualizer
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}
	catch (std::exception& e)
	{
		  std::cerr << "Error inside main " << e.what() << std::endl;
	}
	// cv::waitKey(0);
	int64 elapse_t2 = cv::getTickCount();
	double dTotalTime = (elapse_t2 - elapse_t1) / cv::getTickFrequency();
	std:cout << "Total Time  : " << dTotalTime << std::endl;
	return 0;
}
