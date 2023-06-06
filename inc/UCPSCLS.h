#ifndef	UCPSCLS_HEADER
#define UCPSCLS_HEADER

#include "UCPSCommon.h"

#define IMAGE_MAX_NUMBER			4
#define STRING_MAX_LENGTH			260

class clsUCPS
{
public:	
	clsUCPS(){}
	~clsUCPS(){}

	template <typename T>
	int sgn(T val) {return (T(0) < val) - (val < T(0));}
	cv::Mat				imageMask(std::vector<cv::Mat> camImages);
	cv::Mat				lightPattern(int width, int height, int j, int N);
	cv::Mat				cvtFloatToGrayscale(cv::Mat F, int limit = 255);
	void				exportOBJ(cv::Mat Depth, cv::Mat Normals, cv::Mat texture);	
	void				exportPLY(int width, int height, cv::Mat Z);
	std::string 		type2str(int type);
	
	virtual cv::Mat		localHeightfield(cv::Mat Normals) { return cv::Mat(); }
	virtual cv::Mat		computeNormals(std::vector<cv::Mat> camImages, cv::Mat Mask) { return cv::Mat(); }
	virtual void		updateHeights(cv::Mat &Normals, cv::Mat &Z, int iterations) {}
private:
};

class clsUCPSD1:public clsUCPS
{
public:
	clsUCPSD1(){}
	~clsUCPSD1(){}
	
	cv::Mat		localHeightfield(cv::Mat Normals) override;
	cv::Mat		computeNormals(std::vector<cv::Mat> camImages, cv::Mat Mask) override;
	void		updateHeights(cv::Mat &Normals, cv::Mat &Z, int iterations) override;
private:	
};
#endif
