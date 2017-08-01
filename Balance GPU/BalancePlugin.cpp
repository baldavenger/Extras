#include "BalancePlugin.h"

#include <stdio.h>
#include <cmath>
#include <cfloat> // DBL_MAX
#include <climits>
#include <algorithm>
#include <limits>


#include "ofxsProcessing.h"
#include "ofxsRectangleInteract.h"
#include "ofxsMacros.h"
#include "ofxsCopier.h"
#include "ofxsCoords.h"
#include "ofxsLut.h"
#include "ofxsMultiThread.h"
#include "ofxsLog.h"
#include "ofxsThreadSuite.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#include <windows.h>
#define isnan _isnan
#else
using std::isnan;
#endif

using namespace OFX;

#define kPluginName "Balance"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
	"White Balance Image: use eyedropper to sample pixel RGB values, or use mean values. " \
    "Compute image statistics over the whole image or over a rectangle. " \
    "The statistics can be computed either on RGBA components, in the HSV colorspace " \
    "(which is the HSV colorspace with an additional L component from HSL), or the " \
    "position and value of the pixels with the maximum and minimum luminance values can be computed.\n" \
    "The color values of the minimum and maximum luma pixels for an image sequence "
#define kPluginIdentifier "OpenFX.Yo.Balance"
#define kPluginVersionMajor 2 
#define kPluginVersionMinor 2 

#define kSupportsTiles 1
#define kSupportsMultiResolution 1
#define kSupportsRenderScale 0 // no renderscale support: statistics are computed at full resolution
#define kSupportsMultipleClipPARs false
#define kSupportsMultipleClipDepths false
#define kRenderThreadSafety eRenderFullySafe


#define kParamRestrictToRectangle "restrictToRectangle"
#define kParamRestrictToRectangleLabel "Restrict to Rectangle"
#define kParamRestrictToRectangleHint "Restrict statistics computation to a rectangle."

#define kParamSampleType "Sample Type"
#define kParamSampleTypeLabel "Sample Type"
#define kParamSampleTypeHint "ColorPicker or Mean Average"
#define kParamSampleTypeOptionColorPicker "ColorPicker"
#define kParamSampleTypeOptionColorPickerHint "Use ColorPicker Sample Tool"
#define kParamSampleTypeOptionMean "Mean Average"
#define kParamSampleTypeOptionMeanHint "Use Mean Average"

enum SampleTypeEnum
{
    eSampleTypeColorPicker,
    eSampleTypeMean,
};

#define kParamBalanceType "Balance Type"
#define kParamBalanceTypeLabel "Balance Type"
#define kParamBalanceTypeHint "White Balance Formula"
#define kParamBalanceTypeOptionGain "Gain"
#define kParamBalanceTypeOptionGainHint "Use Gain Formula"
#define kParamBalanceTypeOptionOffset "Offset"
#define kParamBalanceTypeOptionOffsetHint "Use Offset Formula"
#define kParamBalanceTypeOptionLift "Lift"
#define kParamBalanceTypeOptionLiftHint "Use Lift Formula"

enum BalanceTypeEnum
{
    eBalanceTypeGain,
    eBalanceTypeOffset,
    eBalanceTypeLift,
};

#define kParamAnalyzeFrame "analyzeFrame"
#define kParamAnalyzeFrameLabel "Analyze Frame"
#define kParamAnalyzeFrameHint "Analyze current frame and set values."

#define kParamClearFrame "clearFrame"
#define kParamClearFrameLabel "Clear Frame"
#define kParamClearFrameHint "Clear analysis for current frame."

#define kParamGroupRGB "RGB"

#define kParamStatMin "statMin"
#define kParamStatMinLabel "Min."
#define kParamStatMinHint "Minimum value."

#define kParamStatMax "statMax"
#define kParamStatMaxLabel "Max."
#define kParamStatMaxHint "Maximum value."

#define kParamStatMean "statMean"
#define kParamStatMeanLabel "Mean"
#define kParamStatMeanHint "The mean is the average. Add up the values, and divide by the number of values."

//#define kParamStatMedian "statMedian"
//#define kParamStatMedianLabel "Median"
//#define kParamStatMedianHint "The median average. The middle value."

#define kParamGroupHSV "HSV"

#define kParamAnalyzeFrameHSV "analyzeFrameHSV"
#define kParamAnalyzeFrameHSVLabel "Analyze Frame"
#define kParamAnalyzeFrameHSVHint "Analyze current frame as HSV and set values."

#define kParamClearFrameHSV "clearFrameHSV"
#define kParamClearFrameHSVLabel "Clear Frame"
#define kParamClearFrameHSVHint "Clear HSV analysis for current frame."

#define kParamStatHSVMin "statHSVMin"
#define kParamStatHSVMinLabel "HSV Min."
#define kParamStatHSVMinHint "Minimum value."

#define kParamStatHSVMax "statHSVMax"
#define kParamStatHSVMaxLabel "HSV Max."
#define kParamStatHSVMaxHint "Maximum value."

#define kParamStatHSVMean "statHSVMean"
#define kParamStatHSVMeanLabel "HSV Mean"
#define kParamStatHSVMeanHint "The mean is the average. Add up the values, and divide by the number of values."

#define kParamGroupLuma "Min/Max Luma"

#define kParamAnalyzeFrameLuma "analyzeFrameLuma"
#define kParamAnalyzeFrameLumaLabel "Analyze Frame"
#define kParamAnalyzeFrameLumaHint "Analyze current frame and set min/max luma values."

#define kParamClearFrameLuma "clearFrameLuma"
#define kParamClearFrameLumaLabel "Clear Frame"
#define kParamClearFrameLumaHint "Clear luma analysis for current frame."

#define kParamLuminanceMath "luminanceMath"
#define kParamLuminanceMathLabel "Luminance Math"
#define kParamLuminanceMathHint "Formula used to compute luminance from RGB values."
#define kParamLuminanceMathOptionRec709 "Rec. 709"
#define kParamLuminanceMathOptionRec709Hint "Use Rec. 709 (0.2126r + 0.7152g + 0.0722b)."
#define kParamLuminanceMathOptionAverage "Average"
#define kParamLuminanceMathOptionAverageHint "Use average of r, g, b."
#define kParamLuminanceMathOptionMaximum "Max"
#define kParamLuminanceMathOptionMaximumHint "Use max or r, g, b."

enum LuminanceMathEnum
{
    eLuminanceMathRec709,
    eLuminanceMathAverage,
    eLuminanceMathMaximum,
};

#define kParamMaxLumaPix "maxLumaPix"
#define kParamMaxLumaPixLabel "Max Luma Pixel"
#define kParamMaxLumaPixHint "Position of the pixel with the maximum luma value."
#define kParamMaxLumaPixVal "maxLumaPixVal"
#define kParamMaxLumaPixValLabel "Max Luma Pixel Value"
#define kParamMaxLumaPixValHint "RGB value for the pixel with the maximum luma value."

#define kParamMinLumaPix "minLumaPix"
#define kParamMinLumaPixLabel "Min Luma Pixel"
#define kParamMinLumaPixHint "Position of the pixel with the minimum luma value."
#define kParamMinLumaPixVal "minLumaPixVal"
#define kParamMinLumaPixValLabel "Min Luma Pixel Value"
#define kParamMinLumaPixValHint "RGB value for the pixel with the minimum luma value."

#define kParamDefaultsNormalised "defaultsNormalised"

static bool gHostSupportsDefaultCoordinateSystem = true; // for kParamDefaultsNormalised

#define POINT_TOLERANCE 6
#define POINT_SIZE 5

#define kOfxFlagInfiniteMax INT_MAX
#define kOfxFlagInfiniteMin INT_MIN

struct RGBValues
{
    float r, g, b;
    RGBValues(float v) : r(v), g(v), b(v) {}

    RGBValues() : r(0), g(0), b(0) {}
};

struct RGBdub
{
    double r, g, b;
    RGBdub(double v) : r(v), g(v), b(v) {}

    RGBdub() : r(0), g(0), b(0) {}
};

struct Results
{
    Results()
    : MIN( std::numeric_limits<float>::infinity() )
    , MAX( -std::numeric_limits<float>::infinity() )
    , mean(0.)
    , maxVal( -std::numeric_limits<float>::infinity() )
    , minVal( std::numeric_limits<float>::infinity() )
    {
        maxPos.x = maxPos.y = minPos.x = minPos.y = 0.;
    }

    RGBValues MIN;
    RGBValues MAX;
    RGBValues mean;
    OfxPointD maxPos; // luma only
    RGBValues maxVal; // luma only
    OfxPointD minPos; // luma only
    RGBValues minVal; // luma only
};

class BalanceProcessorBase
    : public ImageProcessor
{
protected:
	//MultiThread::Mutex _mutex; //< this is used so we can multi-thread the analysis and protect the shared results
    unsigned long _count;
	
public:
    BalanceProcessorBase(ImageEffect &instance)
        : ImageProcessor(instance)
        //, _mutex()
        , _count(0)
    {
    }


    virtual ~BalanceProcessorBase()
    {
    }

    virtual void setPrevResults(float time, const Results &results) = 0;
    virtual void getResults(Results *results) = 0;
    

protected:

    template<class PIX, int nComponents, int maxValue>
    void toRGB(const PIX *p,
                RGBValues* rgb)
    {
    
    	if ( (nComponents == 4) || (nComponents == 3) ) {
            float r, g, b;
            rgb->r = p[0];
            rgb->g = p[1];
            rgb->b = p[2];
            
        } else {
            rgb->r = 0.;
            rgb->g = 0.;
            rgb->b = 0.;
        }
    }

    template<class PIX, int nComponents, int maxValue>
    void pixToHSV(const PIX *p,
                   float hsv[3])
    {
        if ( (nComponents == 4) || (nComponents == 3) ) {
            float r, g, b;
            r = p[0];
            g = p[1];
            b = p[2];
            Color::rgb_to_hsv(r, g, b, &hsv[0], &hsv[1], &hsv[2]);
            hsv[0] *= 360 / OFXS_HUE_CIRCLE;
            float MIN = std::min(std::min(r, g), b);
            float MAX = std::max(std::max(r, g), b);
        } else {
            hsv[0] = hsv[1] = hsv[2] = 0.0f;
        }
    }

    template<class PIX, int nComponents, int maxValue>
    void toComponents(const RGBValues& rgb,
                      PIX *p)
    {
        if (nComponents == 4) {
            p[0] = rgb.r;
            p[1] = rgb.g;
            p[2] = rgb.b;
        } else if (nComponents == 3) {
            p[0] = rgb.r;
            p[1] = rgb.g;
            p[2] = rgb.b;
        } 
    }
};

class ImageScaler : public ImageProcessor
{
public:
    explicit ImageScaler(ImageEffect &instance);

    virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI procWindow);

    void setSrcImg(Image* p_SrcImg);
    void setScales(float rGain, float bGain, float rOffset, float bOffset, float rLift, 
    float bLift, float lumaMath, float lumaLimit, float GainBalance, float OffsetBalance, 
    float WhiteBalance, float PreserveLuma, float DisplayAlpha, float LumaRec, float LumaAvg);

private:
    Image* _srcImg;
    float _balGain[2];
    float _balOffset[2];
    float _balLift[2];
    float _lumaMath[1];
    float _lumaLimit[1];
    float _GainBalance[1];
    float _OffsetBalance[1];
    float _WhiteBalance[1];
    float _PreserveLuma[1];
    float _DisplayAlpha[1];
    float _LumaRec[1];
    float _LumaAvg[1];
    
};

ImageScaler::ImageScaler(ImageEffect& instance)
	: ImageProcessor(instance)
{
}

extern void RunCudaKernel(int p_Width, int p_Height, float* balGain, float* balOffset, 
float* balLift, float* lumaMath, float* lumaLimit, float* GainBalance, float* OffsetBalance, float* WhiteBalance, 
float* PreserveLuma, float* DisplayAlpha, float* LumaRec, float* LumaAvg, const float* p_Input, float* p_Output);

void ImageScaler::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

	RunCudaKernel(width, height, _balGain, _balOffset, _balLift, _lumaMath, 
    _lumaLimit, _GainBalance, _OffsetBalance, _WhiteBalance, _PreserveLuma, _DisplayAlpha, _LumaRec, _LumaAvg, input, output);
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* balGain, float* balOffset, 
float* balLift, float* lumaMath, float* lumaLimit, float* GainBalance, float* OffsetBalance, float* WhiteBalance, 
float* PreserveLuma, float* DisplayAlpha, float* LumaRec, float* LumaAvg, const float* p_Input, float* p_Output);

void ImageScaler::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());


    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _balGain, _balOffset, _balLift, _lumaMath, 
    _lumaLimit, _GainBalance, _OffsetBalance, _WhiteBalance, _PreserveLuma, _DisplayAlpha, _LumaRec, _LumaAvg, input, output);
}

void ImageScaler::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            // do we have a source image to scale up
            if (srcPix)
            {
            		
			float lumaRec = srcPix[0] * 0.2126f + srcPix[1] * 0.7152f + srcPix[2] * 0.0722f;
			float lumaAvg = (srcPix[0] + srcPix[1] + srcPix[2]) / 3.0f;
			float lumaMax = fmax(fmax(srcPix[0], srcPix[1]), srcPix[2]);
			float luma = _LumaRec[0] == 1.0f ? lumaRec : _LumaAvg[0] == 1.0f ? lumaAvg : lumaMax;
			
			float alpha = _lumaLimit[0] > 1.0f ? luma + (1.0f - _lumaLimit[0]) * (1.0f - luma) : _lumaLimit[0] >= 0.0f ? (luma >= _lumaLimit[0] ? 
			1.0f : luma / _lumaLimit[0]) : _lumaLimit[0] < -1.0f ? (1.0f - luma) + (_lumaLimit[0] + 1.0f) * luma : luma <= (1.0f + _lumaLimit[0]) ? 1.0f : 
			(1.0f - luma) / (1.0f - (_lumaLimit[0] + 1.0f));
			float Alpha = alpha > 1.0f ? 1.0f : alpha < 0.0f ? 0.0f : alpha;
			
			float BalR = _GainBalance[0] == 1.0f ? srcPix[0] * _balGain[0] : _OffsetBalance[0] == 1.0f ? srcPix[0] + _balOffset[0] : srcPix[0] + (_balLift[0] * (1.0f - srcPix[0]));
			float BalB = _GainBalance[0] == 1.0f ? srcPix[2] * _balGain[1] : _OffsetBalance[0] == 1.0f ? srcPix[2] + _balOffset[1] : srcPix[2] + (_balLift[1] * (1.0f - srcPix[2]));
			float Red = _WhiteBalance[0] ? ( _PreserveLuma[0] ? BalR * _lumaMath[0] : BalR) : srcPix[0];
			float Green = _PreserveLuma[0] ? srcPix[1] * _lumaMath[0] : srcPix[1]; 
			float Blue = _WhiteBalance[0] ? ( _PreserveLuma[0] ? BalB * _lumaMath[0] : BalB) : srcPix[2];
			
			dstPix[0] = _DisplayAlpha[0] == 1.0f ? Alpha : Red * Alpha + srcPix[0] * (1.0f - Alpha);
			dstPix[1] = _DisplayAlpha[0] == 1.0f ? Alpha : Green * Alpha + srcPix[1] * (1.0f - Alpha);
			dstPix[2] = _DisplayAlpha[0] == 1.0f ? Alpha : Blue * Alpha + srcPix[2] * (1.0f - Alpha);
			dstPix[3] = _DisplayAlpha[0] == 1.0f ? srcPix[3] : Alpha;
			
            }
            else
            {
                // no src pixel here, be black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }

            // increment the dst pixel
            dstPix += 4;
        }
    }
}

void ImageScaler::setSrcImg(Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void ImageScaler::setScales(float rGain, float bGain, float rOffset, float bOffset, float rLift, 
    float bLift, float lumaMath, float lumaLimit, float GainBalance, float OffsetBalance,
    float WhiteBalance, float PreserveLuma, float DisplayAlpha, float LumaRec, float LumaAvg)
{
    _balGain[0] = rGain;
    _balGain[1] = bGain;
    _balOffset[0] = rOffset;
    _balOffset[1] = bOffset;
    _balLift[0] = rLift;
    _balLift[1] = bLift;
    _lumaMath[0] = lumaMath;
    _lumaLimit[0] = lumaLimit;
    _GainBalance[0] = GainBalance;
    _OffsetBalance[0] = OffsetBalance;
    _WhiteBalance[0] = WhiteBalance;
    _PreserveLuma[0] = PreserveLuma;
    _DisplayAlpha[0] = DisplayAlpha;
    _LumaRec[0] = LumaRec;
    _LumaAvg[0] = LumaAvg;
}

template <class PIX, int nComponents, int maxValue>
class ImageMinMaxMeanProcessor
    : public BalanceProcessorBase
{
private:
    float _MIN[nComponents];
    float _MAX[nComponents];
    float _sum[nComponents];

public:
    ImageMinMaxMeanProcessor(ImageEffect &instance)
        : BalanceProcessorBase(instance)
    {
        std::fill( _MIN, _MIN + nComponents, +std::numeric_limits<float>::infinity() );
        std::fill( _MAX, _MAX + nComponents, -std::numeric_limits<float>::infinity() );
        std::fill(_sum, _sum + nComponents, 0.);
    }

    ~ImageMinMaxMeanProcessor()
    {
    }

    void setPrevResults(float /* time */,
                        const Results & /*results*/) OVERRIDE FINAL {}

    void getResults(Results *results) OVERRIDE FINAL
    {
        if (_count > 0) {
            toRGB<float, nComponents, 1>(_MIN, &results->MIN);
            toRGB<float, nComponents, 1>(_MAX, &results->MAX);
            float mean[nComponents];
            for (int c = 0; c < nComponents; ++c) {
                mean[c] = _sum[c] / _count;
            }
            toRGB<float, nComponents, 1>(mean, &results->mean);
        }
    }

private:

    void addResults(float MIN[nComponents],
                    float MAX[nComponents],
                    float sum[nComponents],
                    unsigned long count)
    {
        //_mutex.lock();
        for (int c = 0; c < nComponents; ++c) {
            _MIN[c] = (std::min(_MIN[c], MIN[c])) / (float)maxValue;
            _MAX[c] = (std::max(_MAX[c], MAX[c])) / (float)maxValue;
            _sum[c] += sum[c] / (float)maxValue;
        }
        _count += count;
        //_mutex.unlock();
    }

    void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
    {
        float MIN[nComponents], MAX[nComponents], sum[nComponents];

        std::fill( MIN, MIN + nComponents, +std::numeric_limits<float>::infinity() );
        std::fill( MAX, MAX + nComponents, -std::numeric_limits<float>::infinity() );
        std::fill(sum, sum + nComponents, 0.);
        unsigned long count = 0;

        assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
               _dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
        for (int y = procWindow.y1; y < procWindow.y2; ++y) {
            if ( _effect.abort() ) {
                break;
            }

            PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);
            float sumLine[nComponents]; // partial sum to avoid underflows
            std::fill(sumLine, sumLine + nComponents, 0.);

            for (int x = procWindow.x1; x < procWindow.x2; ++x) {
                for (int c = 0; c < nComponents; ++c) {
                    float v = *dstPix;
                    MIN[c] = std::min(MIN[c], v);
                    MAX[c] = std::max(MAX[c], v);
                    sumLine[c] += v;
                    ++dstPix;
                }
            }
            for (int c = 0; c < nComponents; ++c) {
                sum[c] += sumLine[c];
            }
            count += procWindow.x2 - procWindow.x1;
        }

        addResults(MIN, MAX, sum, count);
    }
};

#define nComponentsHSV 3

template <class PIX, int nComponents, int maxValue>
class ImageHSVMinMaxMeanProcessor
    : public BalanceProcessorBase
{
private:
    float _MIN[nComponentsHSV];
    float _MAX[nComponentsHSV];
    float _sum[nComponentsHSV];

public:
    ImageHSVMinMaxMeanProcessor(ImageEffect &instance)
        : BalanceProcessorBase(instance)
    {
        std::fill( _MIN, _MIN + nComponentsHSV, +std::numeric_limits<float>::infinity() );
        std::fill( _MAX, _MAX + nComponentsHSV, -std::numeric_limits<float>::infinity() );
        std::fill(_sum, _sum + nComponentsHSV, 0.);
    }

    ~ImageHSVMinMaxMeanProcessor()
    {
    }

    void setPrevResults(float /* time */,
                        const Results & /*results*/) OVERRIDE FINAL {}

    void getResults(Results *results) OVERRIDE FINAL
    {
        if (_count > 0) {
            toRGB<float, nComponentsHSV, 1>(_MIN, &results->MIN);
            toRGB<float, nComponentsHSV, 1>(_MAX, &results->MAX);
            float mean[nComponentsHSV];
            for (int c = 0; c < nComponentsHSV; ++c) {
                mean[c] = _sum[c] / _count;
            }
            toRGB<float, nComponentsHSV, 1>(mean, &results->mean);
        }
    }

private:

    void addResults(float MIN[nComponentsHSV],
                    float MAX[nComponentsHSV],
                    float sum[nComponentsHSV],
                    unsigned long count)
    {
        //_mutex.lock();
        for (int c = 0; c < nComponentsHSV - 1; ++c) {
            _MIN[c] = (std::min(_MIN[c], MIN[c]));
            _MAX[c] = (std::max(_MAX[c], MAX[c]));
            _sum[c] += sum[c];
        }
        _MIN[2] = (std::min(_MIN[2], MIN[2])) / (float)maxValue;
        _MAX[2] = (std::max(_MAX[2], MAX[2])) / (float)maxValue;
        _sum[2] += sum[2] / (float)maxValue;
        _count += count;
        //_mutex.unlock();
    }

    void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
    {
        float MIN[nComponentsHSV], MAX[nComponentsHSV], sum[nComponentsHSV];

        std::fill( MIN, MIN + nComponentsHSV, +std::numeric_limits<float>::infinity() );
        std::fill( MAX, MAX + nComponentsHSV, -std::numeric_limits<float>::infinity() );
        std::fill(sum, sum + nComponentsHSV, 0.);
        unsigned long count = 0;

        assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
               _dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
        for (int y = procWindow.y1; y < procWindow.y2; ++y) {
            if ( _effect.abort() ) {
                break;
            }

            PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);
            float sumLine[nComponentsHSV]; // partial sum to avoid underflows
            std::fill(sumLine, sumLine + nComponentsHSV, 0.);

            for (int x = procWindow.x1; x < procWindow.x2; ++x) {
                float hsv[nComponentsHSV];
                pixToHSV<PIX, nComponents, maxValue>(dstPix, hsv);
                for (int c = 0; c < nComponentsHSV; ++c) {
                    float v = hsv[c];
                    MIN[c] = std::min(MIN[c], v);
                    MAX[c] = std::max(MAX[c], v);
                    sumLine[c] += v;
                }
                dstPix += nComponents;
            }
            for (int c = 0; c < nComponentsHSV; ++c) {
                sum[c] += sumLine[c];
            }
            count += procWindow.x2 - procWindow.x1;
        }

        addResults(MIN, MAX, sum, count);
    }
};

template <class PIX, int nComponents, int maxValue>
class ImageLumaProcessor
    : public BalanceProcessorBase
{
private:
    OfxPointD _maxPos;
    float _maxVal[nComponents];
    float _maxLuma;
    OfxPointD _minPos;
    float _minVal[nComponents];
    float _minLuma;
    LuminanceMathEnum _luminanceMath;

public:
    ImageLumaProcessor(ImageEffect &instance)
        : BalanceProcessorBase(instance)
        , _luminanceMath(eLuminanceMathRec709)
    {
        _maxPos.x = _maxPos.y = 0.;
        std::fill( _maxVal, _maxVal + nComponents, -std::numeric_limits<float>::infinity() );
        _maxLuma = -std::numeric_limits<float>::infinity();
        _minPos.x = _minPos.y = 0.;
        std::fill( _minVal, _minVal + nComponents, +std::numeric_limits<float>::infinity() );
        _minLuma = +std::numeric_limits<float>::infinity();
    }

    ImageLumaProcessor()
    {
    }

    void setPrevResults(float time,
                        const Results & /*results*/) OVERRIDE FINAL
    {
        ChoiceParam* luminanceMath = _effect.fetchChoiceParam(kParamLuminanceMath);

        assert(luminanceMath);
        
        int luma;
        luminanceMath->getValueAtTime(time, luma);
        _luminanceMath = (LuminanceMathEnum)luma;
        
    }

    void getResults(Results *results) OVERRIDE FINAL
    {
        results->maxPos = _maxPos;
        toRGB<float, nComponents, 1>(_maxVal, &results->maxVal);
        results->minPos = _minPos;
        toRGB<float, nComponents, 1>(_minVal, &results->minVal);
    }

private:

    float luminance (const PIX *p)
    {
        if ( (nComponents == 4) || (nComponents == 3) ) {
            float r, g, b;
            r = p[0];
            g = p[1];
            b = p[2];
            switch (_luminanceMath) {
            case eLuminanceMathRec709:
            default:

                return Color::rgb709_to_y(r, g, b);
            case eLuminanceMathAverage:

                return (r + g + b) / 3;
            case eLuminanceMathMaximum:

                return std::fmax(std::fmax(r, g), b);
            }
        }

        return 0.;
    }

    void addResults(const OfxPointD& maxPos,
                    float maxVal[nComponents],
                    float maxLuma,
                    const OfxPointD& minPos,
                    float minVal[nComponents],
                    float minLuma)
    {
        //_mutex.lock();
        if (maxLuma > _maxLuma) {
            _maxPos = maxPos;
            for (int c = 0; c < nComponents; ++c) {
                _maxVal[c] = maxVal[c];
            }
            _maxLuma = maxLuma;
        }
        if (minLuma < _minLuma) {
            _minPos = minPos;
            for (int c = 0; c < nComponents; ++c) {
                _minVal[c] = minVal[c];
            }
            _minLuma = minLuma;
        }
        //_mutex.unlock();
    }

    void multiThreadProcessImages(OfxRectI procWindow) OVERRIDE FINAL
    {
        OfxPointD maxPos = {0., 0.};
        float maxVal[nComponents] = {0.};
        float maxLuma = -std::numeric_limits<float>::infinity();
        OfxPointD minPos = {0., 0.};
        float minVal[nComponents] = {0.};
        float minLuma = +std::numeric_limits<float>::infinity();

        assert(_dstImg->getBounds().x1 <= procWindow.x1 && procWindow.y2 <= _dstImg->getBounds().y2 &&
               _dstImg->getBounds().y1 <= procWindow.y1 && procWindow.y2 <= _dstImg->getBounds().y2);
        for (int y = procWindow.y1; y < procWindow.y2; ++y) {
            if ( _effect.abort() ) {
                break;
            }

            PIX *dstPix = (PIX *) _dstImg->getPixelAddress(procWindow.x1, y);

            for (int x = procWindow.x1; x < procWindow.x2; ++x) {
                float luma = luminance(dstPix);

                if (luma > maxLuma) {
                    maxPos.x = x;
                    maxPos.y = y;
                    for (int c = 0; c < nComponents; ++c) {
                        maxVal[c] = dstPix[c] / (float)maxValue;
                    }
                    maxLuma = luma;
                }
                if (luma < minLuma) {
                    minPos.x = x;
                    minPos.y = y;
                    for (int c = 0; c < nComponents; ++c) {
                        minVal[c] = dstPix[c] / (float)maxValue;
                    }
                    minLuma = luma;
                }

                dstPix += nComponents;
            }
        }

        addResults(maxPos, maxVal, maxLuma, minPos, minVal, minLuma);
    }
};



////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class BalancePlugin
    : public ImageEffect
{
public:
    /** @brief ctor */
    BalancePlugin(OfxImageEffectHandle handle)
        : ImageEffect(handle)
        , _dstClip(0)
        , _srcClip(0)
        , _btmLeft(0)
        , _size(0)
        , _restrictToRectangle(0)
    {
        _dstClip = fetchClip(kOfxImageEffectOutputClipName);
        assert( _dstClip && (!_dstClip->isConnected() || _dstClip->getPixelComponents() == ePixelComponentAlpha ||
                             _dstClip->getPixelComponents() == ePixelComponentRGB ||
                             _dstClip->getPixelComponents() == ePixelComponentRGBA) );
        _srcClip = getContext() == eContextGenerator ? NULL : fetchClip(kOfxImageEffectSimpleSourceClipName);
        assert( (!_srcClip && getContext() == eContextGenerator) ||
                ( _srcClip && (!_srcClip->isConnected() || _srcClip->getPixelComponents() ==  ePixelComponentAlpha ||
                               _srcClip->getPixelComponents() == ePixelComponentRGB ||
                               _srcClip->getPixelComponents() == ePixelComponentRGBA) ) );
                               
        m_Balance = fetchRGBParam("balance");
    	m_Rgb = fetchDouble3DParam("rgbVal");
    	m_Hsl = fetchDouble3DParam("hslVal");
    	m_White = fetchBooleanParam("whiteBalance");
    	_sampleType = fetchChoiceParam(kParamSampleType);
    	_balanceType = fetchChoiceParam(kParamBalanceType);
    	_preserveLuma = fetchBooleanParam("preserveLuma");
    	_lumaLimiter = fetchDoubleParam("lumaLimiter");
    	_displayAlpha = fetchBooleanParam("displayAlpha");

        _btmLeft = fetchDouble2DParam(kParamRectangleInteractBtmLeft);
        _size = fetchDouble2DParam(kParamRectangleInteractSize);
        _restrictToRectangle = fetchBooleanParam(kParamRestrictToRectangle);
        assert(_btmLeft && _size && _restrictToRectangle);
        _statMin = fetchDouble3DParam(kParamStatMin);
        _statMax = fetchDouble3DParam(kParamStatMax);
        _statMean = fetchDouble3DParam(kParamStatMean);
        //_statMedian = fetchDouble3DParam(kParamStatMedian);
        assert(_statMin && _statMax && _statMean);// && _statMedian);
        _analyzeFrame = fetchPushButtonParam(kParamAnalyzeFrame);
        assert(_analyzeFrame);
        _statHSVMin = fetchDouble3DParam(kParamStatHSVMin);
        _statHSVMax = fetchDouble3DParam(kParamStatHSVMax);
        _statHSVMean = fetchDouble3DParam(kParamStatHSVMean);
        assert(_statHSVMin && _statHSVMax && _statHSVMean);
        _analyzeFrameHSV = fetchPushButtonParam(kParamAnalyzeFrameHSV);
        assert(_analyzeFrameHSV);
        _luminanceMath = fetchChoiceParam(kParamLuminanceMath);
        _maxLumaPix = fetchDouble2DParam(kParamMaxLumaPix);
        _maxLumaPixVal = fetchDouble3DParam(kParamMaxLumaPixVal);
        _minLumaPix = fetchDouble2DParam(kParamMinLumaPix);
        _minLumaPixVal = fetchDouble3DParam(kParamMinLumaPixVal);
        assert(_luminanceMath && _maxLumaPix && _maxLumaPixVal && _minLumaPix && _minLumaPixVal);
        // update visibility
        bool restrictToRectangle = _restrictToRectangle->getValue();
        _btmLeft->setIsSecretAndDisabled(!restrictToRectangle);
        _size->setIsSecretAndDisabled(!restrictToRectangle);
        bool WhiteBalance = m_White->getValue();
        _sampleType->setIsSecretAndDisabled(!WhiteBalance);
        _balanceType->setIsSecretAndDisabled(!WhiteBalance);
        _preserveLuma->setIsSecretAndDisabled(!WhiteBalance);

        // honor kParamDefaultsNormalised
        if ( paramExists(kParamDefaultsNormalised) ) {
            // Some hosts (e.g. Resolve) may not support normalized defaults (setDefaultCoordinateSystem(eCoordinatesNormalised))
            // handle these ourselves!
            BooleanParam* param = fetchBooleanParam(kParamDefaultsNormalised);
            assert(param);
            bool normalised = param->getValue();
            if (normalised) {
                OfxPointD size = getProjectExtent();
                OfxPointD origin = getProjectOffset();
                OfxPointD p;
                // we must denormalise all parameters for which setDefaultCoordinateSystem(eCoordinatesNormalised) couldn't be done
                beginEditBlock(kParamDefaultsNormalised);
                _btmLeft->getValue(p.x, p.y);
                _btmLeft->setValue(p.x * size.x + origin.x, p.y * size.y + origin.y);
                _size->getValue(p.x, p.y);
                _size->setValue(p.x * size.x, p.y * size.y);
                param->setValue(false);
                endEditBlock();
            }
        }
    }

private:
    /* override is identity */
    virtual bool isIdentity(const IsIdentityArguments &args, Clip * &identityClip, double &identityTime, int& view, std::string& plane);// OVERRIDE FINAL;


    /* Override the render */
    virtual void render(const RenderArguments &args) OVERRIDE FINAL;
    virtual void getRegionsOfInterest(const RegionsOfInterestArguments &args, RegionOfInterestSetter &rois) OVERRIDE FINAL;
    virtual bool getRegionOfDefinition(const RegionOfDefinitionArguments &args, OfxRectD & rod) OVERRIDE FINAL;
    virtual void changedParam(const InstanceChangedArgs &args, const std::string &paramName) OVERRIDE FINAL;

    /* set up and run a processor */
    void setupAndProcess(ImageScaler &, const RenderArguments &args);
    void YoProcess(BalanceProcessorBase &processor, const Image* srcImg, double time, const OfxRectI &analysisWindow, 
    const Results &prevResults, Results *results);

    // compute computation window in srcImg
    bool computeWindow(const Image* srcImg, double time, OfxRectI *analysisWindow);

    // update image statistics
    void update(const Image* srcImg, double time, const OfxRectI& analysisWindow);
    void updateHSV(const Image* srcImg, double time, const OfxRectI& analysisWindow);
    void updateLuma(const Image* srcImg, double time, const OfxRectI& analysisWindow);

    template <template<class PIX, int nComponents, int maxValue> class Processor, class PIX, int nComponents, int maxValue>
    void updateSubComponentsDepth(const Image* srcImg,
                                  double time,
                                  const OfxRectI &analysisWindow,
                                  const Results& prevResults,
                                  Results* results)

    {
        Processor<PIX, nComponents, maxValue> fred(*this);
        YoProcess(fred, srcImg, time, analysisWindow, prevResults, results);
    }

    template <template<class PIX, int nComponents, int maxValue> class Processor, int nComponents>
    void updateSubComponents(const Image* srcImg,
                             double time,
                             const OfxRectI &analysisWindow,
                             const Results& prevResults,
                             Results* results)
    {
        BitDepthEnum srcBitDepth = srcImg->getPixelDepth();

        switch (srcBitDepth) {
        case eBitDepthUByte: {
            updateSubComponentsDepth<Processor, unsigned char, nComponents, 255>(srcImg, time, analysisWindow, prevResults, results);
            break;
        }
        case eBitDepthUShort: {
            updateSubComponentsDepth<Processor, unsigned short, nComponents, 65535>(srcImg, time, analysisWindow, prevResults, results);
            break;
        }
        case eBitDepthFloat: {
            updateSubComponentsDepth<Processor, float, nComponents, 1>(srcImg, time, analysisWindow, prevResults, results);
            break;
        }
        default:
            throwSuiteStatusException(kOfxStatErrUnsupported);
        }
    }

    template <template<class PIX, int nComponents, int maxValue> class Processor>
    void updateSub(const Image* srcImg,
                   double time,
                   const OfxRectI &analysisWindow,
                   const Results& prevResults,
                   Results* results)
    {
        PixelComponentEnum srcComponents  = srcImg->getPixelComponents();

        assert(srcComponents == ePixelComponentAlpha || srcComponents == ePixelComponentRGB || srcComponents == ePixelComponentRGBA);
        if (srcComponents == ePixelComponentAlpha) {
            updateSubComponents<Processor, 1>(srcImg, time, analysisWindow, prevResults, results);
        } else if (srcComponents == ePixelComponentRGBA) {
            updateSubComponents<Processor, 4>(srcImg, time, analysisWindow, prevResults, results);
        } else if (srcComponents == ePixelComponentRGB) {
            updateSubComponents<Processor, 3>(srcImg, time, analysisWindow, prevResults, results);
        } else {
            // coverity[dead_error_line]
            throwSuiteStatusException(kOfxStatErrUnsupported);
        }
    }

private:

    // do not need to delete these, the ImageEffect is managing them for us
    Clip *_dstClip;
    Clip *_srcClip;
    Double2DParam* _btmLeft;
    Double2DParam* _size;
    ChoiceParam* _sampleType;
    ChoiceParam* _balanceType;
    BooleanParam* _preserveLuma;
    BooleanParam* _displayAlpha;
    DoubleParam* _lumaLimiter;
    BooleanParam* _restrictToRectangle;    
    Double3DParam* _statMin;
    Double3DParam* _statMax;
    Double3DParam* _statMean;
    //Double3DParam* _statMedian;
    PushButtonParam* _analyzeFrame;
    Double3DParam* _statHSVMin;
    Double3DParam* _statHSVMax;
    Double3DParam* _statHSVMean;
    PushButtonParam* _analyzeFrameHSV;
    ChoiceParam* _luminanceMath;
    Double2DParam* _maxLumaPix;
    Double3DParam* _maxLumaPixVal;
    Double2DParam* _minLumaPix;
    Double3DParam* _minLumaPixVal;
    
    RGBParam *m_Balance;
    Double3DParam* m_Rgb;
    Double3DParam* m_Hsl;
    BooleanParam* m_White;
};

////////////////////////////////////////////////////////////////////////////////
/** @brief render for the filter */


// the overridden render function

void
BalancePlugin::setupAndProcess(ImageScaler& p_ImageScaler, const RenderArguments &args)
{
    const double time = args.time;

    std::auto_ptr<OFX::Image> dst( _dstClip->fetchImage(time) );

    if ( !dst.get() ) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }
    OFX::BitDepthEnum dstBitDepth    = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents  = dst->getPixelComponents();
    if ( ( dstBitDepth != _dstClip->getPixelDepth() ) ||
         ( dstComponents != _dstClip->getPixelComponents() ) ) {
        setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong depth or components");
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }
    if ( (dst->getRenderScale().x != args.renderScale.x) ||
         ( dst->getRenderScale().y != args.renderScale.y) ||
         ( ( dst->getField() != OFX::eFieldNone) /* for DaVinci Resolve */ && ( dst->getField() != args.fieldToRender) ) ) {
        setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
        OFX::throwSuiteStatusException(kOfxStatFailed);
    }
    
    std::auto_ptr<OFX::Image> src( ( _srcClip && _srcClip->isConnected() ) ?
                                    _srcClip->fetchImage(time) : 0 );
    if ( src.get() ) {
        if ( (src->getRenderScale().x != args.renderScale.x) ||
             ( src->getRenderScale().y != args.renderScale.y) ||
             ( ( src->getField() != OFX::eFieldNone) /* for DaVinci Resolve */ && ( src->getField() != args.fieldToRender) ) ) {
            setPersistentMessage(OFX::Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
            OFX::throwSuiteStatusException(kOfxStatFailed);
        }
        OFX::BitDepthEnum srcBitDepth      = src->getPixelDepth();
        OFX::PixelComponentEnum srcComponents = src->getPixelComponents();
        // set the components of _dstClip
        if ( (srcBitDepth != dstBitDepth) ||  (srcComponents != dstComponents) )  {
            OFX::throwSuiteStatusException(kOfxStatErrImageFormat);
        }
    }
    
    int sampleType_i;
    _sampleType->getValueAtTime(args.time, sampleType_i);
    SampleTypeEnum sampleType = (SampleTypeEnum)sampleType_i;
    
    bool ColorSample = sampleType_i == 0;
    bool MeanSample = sampleType_i == 1;
    
    int balanceType_i;
    _balanceType->getValueAtTime(args.time, balanceType_i);
    BalanceTypeEnum balanceType = (BalanceTypeEnum)balanceType_i;
    
    int luminanceMath_i;
    _luminanceMath->getValueAtTime(args.time, luminanceMath_i);
    LuminanceMathEnum luminanceMath = (LuminanceMathEnum)luminanceMath_i;
    
    bool RecLuminanceMath = luminanceMath_i == 0;
    bool AvgLuminanceMath = luminanceMath_i == 1;
    bool MaxLuminanceMath = luminanceMath_i == 2;
    
    RGBdub colorSample;
    m_Balance->getValueAtTime(args.time, colorSample.r, colorSample.g, colorSample.b);
    
    RGBdub meanSample;
    _statMean->getValueAtTime(args.time, meanSample.r, meanSample.g, meanSample.b);
    
    float BalanceR = ColorSample ? colorSample.r : meanSample.r;
    float BalanceG = ColorSample ? colorSample.g : meanSample.g;
    float BalanceB = ColorSample ? colorSample.b : meanSample.b;
    
    float rGain = BalanceG/BalanceR;
    float bGain = BalanceG/BalanceB;
    
    float rOffset = BalanceG - BalanceR;
    float bOffset = BalanceG - BalanceB;
    
    float lumaRec = BalanceR * 0.2126f + BalanceG * 0.7152f + BalanceB * 0.0722f;
    float lumaAvg = (BalanceR + BalanceG + BalanceB) / 3.0f;
    float lumaMax = fmax(fmax(BalanceR, BalanceG), BalanceB);
    float lumaMathChoice = RecLuminanceMath ? lumaRec : AvgLuminanceMath ? lumaAvg : lumaMax;
    float lumaMath = lumaMathChoice / BalanceG;
    
    //BalanceR + (rLift * (1.0 - BalanceR)) = BalanceG
    float rLift = (BalanceG - BalanceR) / (1.0f - BalanceR);
    float bLift = (BalanceG - BalanceB) / (1.0f - BalanceB);
    
    bool preserveLuma = _preserveLuma->getValueAtTime(args.time);
    bool displayAlpha = _displayAlpha->getValueAtTime(args.time);
    
    float lumaLimit = _lumaLimiter->getValueAtTime(args.time);

	bool whiteBalance = m_White->getValueAtTime(args.time);
	float WhiteBalance = whiteBalance ? 1.0f : 0.0f;
	float PreserveLuma = preserveLuma ? 1.0f : 0.0f;
	float LumaLimit = lumaLimit ? 1.0f : 0.0f;
	float DisplayAlpha = displayAlpha ? 1.0f : 0.0f;
	
	bool gainBalance = balanceType_i == 0;
    bool offsetBalance = balanceType_i == 1;
    bool liftBalance = balanceType_i == 2;
    
    float GainBalance = gainBalance ? 1.0f : 0.0f;
    float OffsetBalance = offsetBalance ? 1.0f : 0.0f;
    float LiftBalance = liftBalance ? 1.0f : 0.0f;
    
    float LumaRec = RecLuminanceMath ? 1.0f : 0.0f;
    float LumaAvg = AvgLuminanceMath ? 1.0f : 0.0f;
    float LumaMax = MaxLuminanceMath ? 1.0f : 0.0f;
    
    
    p_ImageScaler.setDstImg(dst.get());
    p_ImageScaler.setSrcImg(src.get());
    
    // Setup OpenCL and CUDA Render arguments
    p_ImageScaler.setGPURenderArgs(args);
    
    p_ImageScaler.setRenderWindow(args.renderWindow);
    
    p_ImageScaler.setScales(rGain, bGain, rOffset, bOffset, rLift, bLift, lumaMath, lumaLimit, GainBalance, OffsetBalance, 
    WhiteBalance, PreserveLuma, DisplayAlpha, LumaRec, LumaAvg);

    p_ImageScaler.process();
} 

void
BalancePlugin::render(const RenderArguments &args)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        throwSuiteStatusException(kOfxStatFailed);
    }

    assert( kSupportsMultipleClipPARs   || !_srcClip || _srcClip->getPixelAspectRatio() == _dstClip->getPixelAspectRatio() );
    assert( kSupportsMultipleClipDepths || !_srcClip || _srcClip->getPixelDepth()       == _dstClip->getPixelDepth() );
    // do the rendering
    std::auto_ptr<Image> dst( _dstClip->fetchImage(args.time) );
    if ( !dst.get() ) {
        throwSuiteStatusException(kOfxStatFailed);
    }
    if ( (dst->getRenderScale().x != args.renderScale.x) ||
         ( dst->getRenderScale().y != args.renderScale.y) ||
         ( ( dst->getField() != eFieldNone)  && ( dst->getField() != args.fieldToRender) ) ) {
        setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
        throwSuiteStatusException(kOfxStatFailed);
    }
    BitDepthEnum dstBitDepth       = dst->getPixelDepth();
    PixelComponentEnum dstComponents  = dst->getPixelComponents();
    std::auto_ptr<const Image> src( ( _srcClip && _srcClip->isConnected() ) ?
                                    _srcClip->fetchImage(args.time) : 0 );
    if ( src.get() ) {
        if ( (src->getRenderScale().x != args.renderScale.x) ||
             ( src->getRenderScale().y != args.renderScale.y) ||
             ( ( src->getField() != eFieldNone)  && ( src->getField() != args.fieldToRender) ) ) {
            setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
            throwSuiteStatusException(kOfxStatFailed);
        }
        BitDepthEnum srcBitDepth      = src->getPixelDepth();
        PixelComponentEnum srcComponents = src->getPixelComponents();
        if ( (srcBitDepth != dstBitDepth) || (srcComponents != dstComponents) ) {
            throwSuiteStatusException(kOfxStatErrImageFormat);
        }
    }

	if (dstComponents == ePixelComponentRGBA || ePixelComponentRGB)
        {
            //copyPixels( *this, args.renderWindow, src.get(), dst.get() );
            ImageScaler fred(*this);
            setupAndProcess(fred, args);
         }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }


} // BalancePlugin::render

// override the roi call
// Required if the plugin requires a region from the inputs which is different from the rendered region of the output.
// (this is the case here)
void
BalancePlugin::getRegionsOfInterest(const RegionsOfInterestArguments &args,
                                            RegionOfInterestSetter &rois)
{
    bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);

    if (restrictToRectangle) {
        OfxRectD regionOfInterest;
        _btmLeft->getValueAtTime(args.time, regionOfInterest.x1, regionOfInterest.y1);
        _size->getValueAtTime(args.time, regionOfInterest.x2, regionOfInterest.y2);
        regionOfInterest.x2 += regionOfInterest.x1;
        regionOfInterest.y2 += regionOfInterest.y1;
        // Union with output RoD, so that render works
        Coords::rectBoundingBox(args.regionOfInterest, regionOfInterest, &regionOfInterest);
        rois.setRegionOfInterest(*_srcClip, regionOfInterest);
    }
}

bool
BalancePlugin::getRegionOfDefinition(const RegionOfDefinitionArguments &args,
                                             OfxRectD & /*rod*/)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        throwSuiteStatusException(kOfxStatFailed);
    }

    return false;
}

bool
BalancePlugin::isIdentity(const IsIdentityArguments &args,
                                  Clip * &identityClip,
                                  double & identityTime
                                  , int& /*view*/, std::string& /*plane*/)
{
    
    RGBdub colorSample;
    m_Balance->getValueAtTime(args.time, colorSample.r, colorSample.g, colorSample.b);
    
    
    float rBalance = colorSample.r;
    float gBalance = colorSample.g;
    float bBalance = colorSample.b;
    

    if ((rBalance == 1.0) && (gBalance == 1.0) && (bBalance == 1.0))
    {
        identityClip = _srcClip;
        identityTime = args.time;
        return true;
    }

    return false;
}

void
BalancePlugin::changedParam(const InstanceChangedArgs &args,
                                    const std::string &paramName)
{
    if ( !kSupportsRenderScale && ( (args.renderScale.x != 1.) || (args.renderScale.y != 1.) ) ) {
        throwSuiteStatusException(kOfxStatFailed);
    }

    //bool doUpdate = false;
    bool doAnalyzeRGB = false;
    bool doAnalyzeHSV = false;
    bool doAnalyzeLuma = false;
    OfxRectI analysisWindow;
    const double time = args.time;
    
    if (paramName == "whiteBalance") {
        // update visibility
        bool WhiteBalance = m_White->getValueAtTime(time);
        _sampleType->setIsSecretAndDisabled(!WhiteBalance);
        _balanceType->setIsSecretAndDisabled(!WhiteBalance);
        _preserveLuma->setIsSecretAndDisabled(!WhiteBalance);
    }

    if (paramName == kParamRestrictToRectangle) {
        // update visibility
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(time);
        _btmLeft->setIsSecretAndDisabled(!restrictToRectangle);
        _size->setIsSecretAndDisabled(!restrictToRectangle);
    }
    if (paramName == kParamAnalyzeFrame) {
        doAnalyzeRGB = true;
    }
    if (paramName == kParamAnalyzeFrameHSV) {
        doAnalyzeHSV = true;
    }
    if (paramName == kParamAnalyzeFrameLuma) {
        doAnalyzeLuma = true;
    }
    if (paramName == kParamClearFrame) {
        _statMin->setValue(0.0, 0.0, 0.0);
        _statMax->setValue(1.0, 1.0, 1.0);
        _statMean->setValue(1.0, 1.0, 1.0);
        //_statMedian->setValue(0.0, 0.0, 0.0);
    }
    if (paramName == kParamClearFrameHSV) {
        _statHSVMin->setValue(0.0, 0.0, 0.0);
        _statHSVMax->setValue(1.0, 1.0, 1.0);
        _statHSVMean->setValue(1.0, 1.0, 1.0);
    }
    if (paramName == kParamClearFrameLuma) {
        _maxLumaPix->setValue(0, 0);
        _maxLumaPixVal->setValue(0.0, 0.0, 0.0);
        _minLumaPix->setValue(0, 0);
        _minLumaPixVal->setValue(1.0, 1.0, 1.0);
    }
   
    // RGB analysis
    if ( (doAnalyzeRGB || doAnalyzeHSV || doAnalyzeLuma) && _srcClip && _srcClip->isConnected() ) {
        std::auto_ptr<Image> src( ( _srcClip && _srcClip->isConnected() ) ?
                                  _srcClip->fetchImage(args.time) : 0 );
        if ( src.get() ) {
            if ( (src->getRenderScale().x != args.renderScale.x) ||
                 ( src->getRenderScale().y != args.renderScale.y) ) {
                setPersistentMessage(Message::eMessageError, "", "OFX Host gave image with wrong scale or field properties");
                throwSuiteStatusException(kOfxStatFailed);
            }
            bool intersect = computeWindow(src.get(), args.time, &analysisWindow);
            if (intersect) {
#             ifdef kOfxImageEffectPropInAnalysis // removed from OFX 1.4
                getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 1, false);
#             endif
                beginEditBlock("analyzeFrame");
                if (doAnalyzeRGB) {
                    update(src.get(), args.time, analysisWindow);
                }
                if (doAnalyzeHSV) {
                    updateHSV(src.get(), args.time, analysisWindow);
                }
                if (doAnalyzeLuma) {
                    updateLuma(src.get(), args.time, analysisWindow);
                }
                endEditBlock();
#             ifdef kOfxImageEffectPropInAnalysis // removed from OFX 1.4
                getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 0, false);
#             endif
            }
        }
    }
   
#     ifdef kOfxImageEffectPropInAnalysis // removed from OFX 1.4
        getPropertySet().propSetInt(kOfxImageEffectPropInAnalysis, 0, false);
#     endif


	if (paramName == "balance")
    {
	 RGBdub colorSample;
	 m_Balance->getValueAtTime(args.time, colorSample.r, colorSample.g, colorSample.b);
	 
	 m_Rgb->setValue(colorSample.r, colorSample.g, colorSample.b);
	 
	 float MIN = std::min(colorSample.r, std::min(colorSample.g, colorSample.b));    
	 float MAX = std::max(colorSample.r, std::max(colorSample.g, colorSample.b));    
	 float del_MAX = MAX - MIN;
		
	 float L = (MAX + MIN) / 2.0f;
	 float S = del_MAX == 0.0f ? 0.0f : (L < 0.5f ? del_MAX / (MAX + MIN) : del_MAX / (2.0f - MAX - MIN));

	 float del_R = (((MAX - colorSample.r) / 6.0f) + (del_MAX / 2.0f)) / del_MAX;
	 float del_G = (((MAX - colorSample.g) / 6.0f) + (del_MAX / 2.0f)) / del_MAX;
	 float del_B = (((MAX - colorSample.b) / 6.0f) + (del_MAX / 2.0f)) / del_MAX;

	 float h = del_MAX == 0.0f ? 0.0f : (colorSample.r == MAX ? del_B - del_G : (colorSample.g == MAX ? (1.0f / 3.0f) + del_R - del_B : (2.0f / 3.0f) + del_G - del_R));
	 
	 float H = h < 0.0f ? h + 1.0f : (h > 1.0f ? h - 1.0f : h);
	 
	 m_Hsl->setValue(H, S, L);
  		 
    }

    }
//} // BalancePlugin::changedParam

/* set up and run a processor */
void
BalancePlugin::YoProcess(BalanceProcessorBase &processor,
                                       const Image* srcImg,
                                       double time,
                                       const OfxRectI &analysisWindow,
                                       const Results &prevResults,
                                       Results *results)
{
    // set the images
    processor.setDstImg( const_cast<Image*>(srcImg) ); // not a bug: we only set dst
     
    // set the render window
    processor.setRenderWindow(analysisWindow);

    processor.setPrevResults(time, prevResults);

    // Call the base class process member, this will call the derived templated process code
    processor.process();

    if ( !abort() ) {
        processor.getResults(results);
    }
}

bool
BalancePlugin::computeWindow(const Image* srcImg,
                                     double time,
                                     OfxRectI *analysisWindow)
{
    OfxRectD regionOfInterest;
    bool restrictToRectangle = _restrictToRectangle->getValueAtTime(time);

    if (!restrictToRectangle && _srcClip) {
        //return false; // no analysis in this case
        
        // use the src region of definition as rectangle, but avoid infinite rectangle
        regionOfInterest = _srcClip->getRegionOfDefinition(time);
        OfxPointD size = getProjectSize();
        OfxPointD offset = getProjectOffset();
        if (regionOfInterest.x1 <= kOfxFlagInfiniteMin) {
            regionOfInterest.x1 = offset.x;
        }
        if (regionOfInterest.x2 >= kOfxFlagInfiniteMax) {
            regionOfInterest.x2 = offset.x + size.x;
        }
        if (regionOfInterest.y1 <= kOfxFlagInfiniteMin) {
            regionOfInterest.y1 = offset.y;
        }
        if (regionOfInterest.y2 >= kOfxFlagInfiniteMax) {
            regionOfInterest.y2 = offset.y + size.y;
        }
    } else {
        _btmLeft->getValueAtTime(time, regionOfInterest.x1, regionOfInterest.y1);
        _size->getValueAtTime(time, regionOfInterest.x2, regionOfInterest.y2);
        regionOfInterest.x2 += regionOfInterest.x1;
        regionOfInterest.y2 += regionOfInterest.y1;
    }
    Coords::toPixelEnclosing(regionOfInterest,
                             srcImg->getRenderScale(),
                             srcImg->getPixelAspectRatio(),
                             analysisWindow);

    return Coords::rectIntersection(*analysisWindow, srcImg->getBounds(), analysisWindow);
}

// update image statistics
void
BalancePlugin::update(const Image* srcImg,
                              double time,
                              const OfxRectI &analysisWindow)
{
    // TODO: CHECK if checkDoubleAnalysis param is true and analysisWindow is the same as btmLeft/sizeAnalysis
    Results results;

    if ( !abort() ) {
        updateSub<ImageMinMaxMeanProcessor>(srcImg, time, analysisWindow, results, &results);
    }
    if ( abort() ) {
        return;
    }
    _statMin->setValueAtTime(time, results.MIN.r, results.MIN.g, results.MIN.b);
    _statMax->setValueAtTime(time, results.MAX.r, results.MAX.g, results.MAX.b);
    _statMean->setValueAtTime(time, results.mean.r, results.mean.g, results.mean.b);
    //_statMedian->setValueAtTime(time, results.medR, results.medG, results.medB);
}

void
BalancePlugin::updateHSV(const Image* srcImg,
                                  double time,
                                  const OfxRectI &analysisWindow)
{
    Results results;

    if ( !abort() ) {
        updateSub<ImageHSVMinMaxMeanProcessor>(srcImg, time, analysisWindow, results, &results);
    }
    if ( abort() ) {
        return;
    }
    _statHSVMin->setValueAtTime(time, results.MIN.r, results.MIN.g, results.MIN.b);
    _statHSVMax->setValueAtTime(time, results.MAX.r, results.MAX.g, results.MAX.b);
    _statHSVMean->setValueAtTime(time, results.mean.r, results.mean.g, results.mean.b);
}

void
BalancePlugin::updateLuma(const Image* srcImg,
                                  double time,
                                  const OfxRectI &analysisWindow)
{
    Results results;

    if ( !abort() ) {
        updateSub<ImageLumaProcessor>(srcImg, time, analysisWindow, results, &results);
    }
    if ( abort() ) {
        return;
    }
    _maxLumaPix->setValueAtTime(time, results.maxPos.x, results.maxPos.y);
    _maxLumaPixVal->setValueAtTime(time, results.maxVal.r, results.maxVal.g, results.maxVal.b);
    _minLumaPix->setValueAtTime(time, results.minPos.x, results.minPos.y);
    _minLumaPixVal->setValueAtTime(time, results.minVal.r, results.minVal.g, results.minVal.b);
}

class BalanceInteract
    : public RectangleInteract
{
public:

    BalanceInteract(OfxInteractHandle handle,
                            ImageEffect* effect)
        : RectangleInteract(handle, effect)
        , _restrictToRectangle(0)
    {
        _restrictToRectangle = effect->fetchBooleanParam(kParamRestrictToRectangle);
        addParamToSlaveTo(_restrictToRectangle);
    }

private:

    // overridden functions from Interact to do things
    virtual bool draw(const DrawArgs &args) OVERRIDE FINAL
    {
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);

        if (restrictToRectangle) {
            return RectangleInteract::draw(args);
        }

        return false;
    }

    virtual bool penMotion(const PenArgs &args) OVERRIDE FINAL
    {
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);

        if (restrictToRectangle) {
            return RectangleInteract::penMotion(args);
        }

        return false;
    }

    virtual bool penDown(const PenArgs &args) OVERRIDE FINAL
    {
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);

        if (restrictToRectangle) {
            return RectangleInteract::penDown(args);
        }

        return false;
    }

    virtual bool penUp(const PenArgs &args) OVERRIDE FINAL
    {
        bool restrictToRectangle = _restrictToRectangle->getValueAtTime(args.time);

        if (restrictToRectangle) {
            return RectangleInteract::penUp(args);
        }

        return false;
    }

    //virtual bool keyDown(const KeyArgs &args) OVERRIDE;
    //virtual bool keyUp(const KeyArgs & args) OVERRIDE;
    //virtual void loseFocus(const FocusArgs &args) OVERRIDE FINAL;


    BooleanParam* _restrictToRectangle;
};

class BalanceOverlayDescriptor
    : public DefaultEffectOverlayDescriptor<BalanceOverlayDescriptor, BalanceInteract>
{
};

BalancePluginFactory::BalancePluginFactory()
    : OFX::PluginFactoryHelper<BalancePluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}


void
BalancePluginFactory::describe(ImageEffectDescriptor &desc)
{
    // basic labels
    desc.setLabel(kPluginName);
    desc.setPluginGrouping(kPluginGrouping);
    desc.setPluginDescription(kPluginDescription);

    desc.addSupportedContext(eContextGeneral);
    desc.addSupportedContext(eContextFilter);

    //desc.addSupportedBitDepth(eBitDepthUByte);
    //desc.addSupportedBitDepth(eBitDepthUShort);
    desc.addSupportedBitDepth(eBitDepthFloat);


    desc.setSingleInstance(false);
    desc.setHostFrameThreading(false);
    desc.setSupportsMultiResolution(kSupportsMultiResolution);
    desc.setSupportsTiles(kSupportsTiles);
    desc.setTemporalClipAccess(false);
    desc.setRenderTwiceAlways(false);
    desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);
    desc.setSupportsMultipleClipDepths(kSupportsMultipleClipDepths);
    //desc.setRenderThreadSafety(kRenderThreadSafety);
    
    
    
     // Setup OpenCL and CUDA render capability flags
    desc.setSupportsOpenCLRender(true);
    desc.setSupportsCudaRender(true);

    // in order to support multiresolution, render() must take into account the pixelaspectratio and the renderscale
    // and scale the transform appropriately.
    // All other functions are usually in canonical coordinates.
    
    desc.setOverlayInteractDescriptor(new BalanceOverlayDescriptor);

}


void
BalancePluginFactory::describeInContext(ImageEffectDescriptor &desc,
                                                ContextEnum /*context*/)
{
    // Source clip only in the filter context
    // create the mandated source clip
    // always declare the source clip first, because some hosts may consider
    // it as the default input clip (e.g. Nuke)
    ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);

    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->addSupportedComponent(ePixelComponentRGB);
    srcClip->addSupportedComponent(ePixelComponentAlpha);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);
    srcClip->setOptional(false);

    // create the mandated output clip
    ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentRGB);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    // make some pages and to things in
    PageParamDescriptor *page = desc.definePageParam("Controls");
    
    
{
        RGBParamDescriptor *param = desc.defineRGBParam("balance");
        param->setLabel("Sample Pixel");
        param->setHint("sample pixel RGB value");
        param->setDefault(1.0, 1.0, 1.0);
        param->setRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
        param->setDisplayRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        page->addChild(*param);
    }

	Double3DParamDescriptor* rgbVal = desc.defineDouble3DParam("rgbVal");
    rgbVal->setLabel("RGB Values");
    //pixelVal->setDoubleType(OFX::eDoubleTypeXYAbsolute);
    rgbVal->setDimensionLabels("r", "g", "b");
    rgbVal->setDefault(1.0, 1.0, 1.0);
    //pixelVal->setIncrement(0.0001);
    rgbVal->setDisplayRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
    //pixelVal->setAnimates(true); // can animate
    page->addChild(*rgbVal);
    
    Double3DParamDescriptor* hslVal = desc.defineDouble3DParam("hslVal");
    hslVal->setLabel("HSL Values");
    //pixelVal->setDoubleType(OFX::eDoubleTypeXYAbsolute);
    hslVal->setDimensionLabels("r", "g", "b");
    hslVal->setDefault(0.0, 0.0, 1.0);
    //pixelVal->setIncrement(0.0001);
    hslVal->setDisplayRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
    //pixelVal->setAnimates(true); // can animate
    page->addChild(*hslVal);
    
    BooleanParamDescriptor* boolParam = desc.defineBooleanParam("whiteBalance");
    boolParam->setDefault(false);
    boolParam->setHint("white balance image");
    boolParam->setLabel("White Balance");
    page->addChild(*boolParam);
    
    {
            ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamSampleType);
            param->setLabel(kParamSampleTypeLabel);
            param->setHint(kParamSampleTypeHint);
            assert(param->getNOptions() == eSampleTypeColorPicker);
            param->appendOption(kParamSampleTypeOptionColorPicker, kParamSampleTypeOptionColorPickerHint);
            assert(param->getNOptions() == eSampleTypeMean);
            param->appendOption(kParamSampleTypeOptionMean, kParamSampleTypeOptionMeanHint);
            //assert(param->getNOptions() == eSampleTypeMedian);
            //param->appendOption(kParamSampleTypeOptionMedian, kParamSampleTypeOptionMedianHint);
            //param->setIsSecretAndDisabled(true);
            page->addChild(*param);
        }
        
	{
            ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamBalanceType);
            param->setLabel(kParamBalanceTypeLabel);
            param->setHint(kParamBalanceTypeHint);
            assert(param->getNOptions() == eBalanceTypeGain);
            param->appendOption(kParamBalanceTypeOptionGain, kParamBalanceTypeOptionGainHint);
            assert(param->getNOptions() == eBalanceTypeOffset);
            param->appendOption(kParamBalanceTypeOptionOffset, kParamBalanceTypeOptionOffsetHint);
            assert(param->getNOptions() == eBalanceTypeLift);
            param->appendOption(kParamBalanceTypeOptionLift, kParamBalanceTypeOptionLiftHint);
            //param->setIsSecretAndDisabled(true);
            page->addChild(*param);
        }
    {  
    	BooleanParamDescriptor* boolParam = desc.defineBooleanParam("preserveLuma");
		boolParam->setDefault(false);
		boolParam->setHint("preserve luma value");
		boolParam->setLabel("Preserve Luma");
		//boolParam->setIsSecretAndDisabled(true);
		page->addChild(*boolParam);
		
		}
		
		// Luma Limit
	{
        GroupParamDescriptor* group = desc.defineGroupParam("lumaLimit");
        group->setOpen(false);
        if (group) {
            group->setLabel("Luma Limiter");
            if (page) {
                page->addChild(*group);
            }
        }
        
        {
            DoubleParamDescriptor *param = desc.defineDoubleParam("lumaLimiter");
            param->setLabel("Luma Limiter");
            param->setHint("limit to luma range");
            param->setDefault(0.0);
            param->setRange(-2.0, 2.0);
            param->setDisplayRange(-2.0, 2.0);
            param->setIncrement(0.001);
            param->setDoubleType(eDoubleTypeScale);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
			BooleanParamDescriptor* boolParam = desc.defineBooleanParam("displayAlpha");
			boolParam->setDefault(false);
			boolParam->setHint("display alpha channel");
			boolParam->setLabel("Display Alpha");
			if (group) {
                boolParam->setParent(*group);
            }
            if (page) {
                page->addChild(*boolParam);
            }
		}
    
    }	
	{
        GroupParamDescriptor* group = desc.defineGroupParam("sampleRegion");
        if (group) {
            group->setLabel("Sample Region");
            if (page) {
                page->addChild(*group);
            }
        }
		
    // restrictToRectangle
    {
        BooleanParamDescriptor *param = desc.defineBooleanParam(kParamRestrictToRectangle);
        param->setLabel(kParamRestrictToRectangleLabel);
        param->setHint(kParamRestrictToRectangleHint);
        param->setDefault(true);
        param->setAnimates(false);
        if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
    }

    // btmLeft
    {
        Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamRectangleInteractBtmLeft);
        param->setLabel(kParamRectangleInteractBtmLeftLabel);
        param->setDoubleType(eDoubleTypeXYAbsolute);
        if ( param->supportsDefaultCoordinateSystem() ) {
            param->setDefaultCoordinateSystem(eCoordinatesNormalised); // no need of kParamDefaultsNormalised
        } else {
            gHostSupportsDefaultCoordinateSystem = false; // no multithread here, see kParamDefaultsNormalised
        }
        param->setDefault(860, 440);
        param->setRange(-DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
        param->setDisplayRange(-10000, -10000, 10000, 10000); // Resolve requires display range or values are clamped to (-1,1)
        param->setIncrement(1);
        param->setHint(kParamRectangleInteractBtmLeftHint);
        param->setDigits(0);
        param->setEvaluateOnChange(false);
        param->setAnimates(true);
        if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
    }

    // size
    {
        Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamRectangleInteractSize);
        param->setLabel(kParamRectangleInteractSizeLabel);
        param->setDoubleType(eDoubleTypeXYAbsolute);
        if ( param->supportsDefaultCoordinateSystem() ) {
            param->setDefaultCoordinateSystem(eCoordinatesNormalised); // no need of kParamDefaultsNormalised
        } else {
            gHostSupportsDefaultCoordinateSystem = false; // no multithread here, see kParamDefaultsNormalised
        }
        param->setDefault(200, 200);
        param->setRange(0, 0, DBL_MAX, DBL_MAX); // Resolve requires range and display range or values are clamped to (-1,1)
        param->setDisplayRange(0, 0, 10000, 10000); // Resolve requires display range or values are clamped to (-1,1)
        param->setIncrement(1.);
        param->setDimensionLabels(kParamRectangleInteractSizeDim1, kParamRectangleInteractSizeDim2);
        param->setHint(kParamRectangleInteractSizeHint);
        param->setDigits(0);
        param->setEvaluateOnChange(false);
        param->setAnimates(true);
        if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
    }
}
	
    {
        GroupParamDescriptor* group = desc.defineGroupParam(kParamGroupRGB);
        if (group) {
            group->setLabel(kParamGroupRGB);
            if (page) {
                page->addChild(*group);
            }
        }
        // min
        {
            Double3DParamDescriptor* param = desc.defineDouble3DParam(kParamStatMin);
            param->setLabel(kParamStatMinLabel);
            param->setDimensionLabels("r", "g", "b");
            param->setHint(kParamStatMinHint);
            param->setDefault(0.0, 0.0, 0.0);
			param->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
			param->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
			param->setDoubleType(eDoubleTypeScale);
			param->setIncrement(0.0001);
			param->setDigits(6);
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        // statMax
        {
            Double3DParamDescriptor* param = desc.defineDouble3DParam(kParamStatMax);
            param->setLabel(kParamStatMaxLabel);
            param->setDimensionLabels("r", "g", "b");
            param->setHint(kParamStatMaxHint);
            param->setDefault(1.0, 1.0, 1.0);
			param->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
			param->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
			param->setDoubleType(eDoubleTypeScale);
			param->setIncrement(0.0001);
			param->setDigits(6);
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        // statMean
        {
            Double3DParamDescriptor* param = desc.defineDouble3DParam(kParamStatMean);
            param->setLabel(kParamStatMeanLabel);
            param->setDimensionLabels("r", "g", "b");
            param->setHint(kParamStatMeanHint);
            param->setDefault(1.0, 1.0, 1.0);
			param->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
			param->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
			param->setDoubleType(eDoubleTypeScale);
			param->setIncrement(0.0001);
			param->setDigits(6);
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
		/*
		// statMedian
        {
            Double3DParamDescriptor* param = desc.defineDouble3DParam(kParamStatMedian);
            param->setLabel(kParamStatMedianLabel);
            param->setDimensionLabels("r", "g", "b");
            param->setHint(kParamStatMedianHint);
            param->setDefault(0.0, 0.0, 0.0);
			param->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
			param->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
			param->setDoubleType(eDoubleTypeScale);
			param->setIncrement(0.0001);
			param->setDigits(6);
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
		*/
        // analyzeFrame
        {
            PushButtonParamDescriptor *param = desc.definePushButtonParam(kParamAnalyzeFrame);
            param->setLabel(kParamAnalyzeFrameLabel);
            param->setHint(kParamAnalyzeFrameHint);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        // clearFrame
        {
            PushButtonParamDescriptor *param = desc.definePushButtonParam(kParamClearFrame);
            param->setLabel(kParamClearFrameLabel);
            param->setHint(kParamClearFrameHint);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

    }

    {
        GroupParamDescriptor* group = desc.defineGroupParam(kParamGroupHSV);
        group->setOpen(false);
        if (group) {
            group->setLabel(kParamGroupHSV);
            if (page) {
                page->addChild(*group);
            }
        }

        // min
        {
            Double3DParamDescriptor* param = desc.defineDouble3DParam(kParamStatHSVMin);
            param->setLabel(kParamStatHSVMinLabel);
            param->setHint(kParamStatHSVMinHint);
            param->setDefault(0.0, 0.0, 0.0);
			param->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
			param->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
			param->setDoubleType(eDoubleTypeScale);
			param->setIncrement(0.0001);
			param->setDigits(6);
            param->setDimensionLabels("h", "s", "v");
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        // statHSVMax
        {
            Double3DParamDescriptor* param = desc.defineDouble3DParam(kParamStatHSVMax);
            param->setLabel(kParamStatHSVMaxLabel);
            param->setHint(kParamStatHSVMaxHint);
            param->setDefault(1.0, 1.0, 1.0);
			param->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
			param->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
			param->setDoubleType(eDoubleTypeScale);
			param->setIncrement(0.0001);
			param->setDigits(6);
            param->setDimensionLabels("h", "s", "v");
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        // statHSVMean
        {
            Double3DParamDescriptor* param = desc.defineDouble3DParam(kParamStatHSVMean);
            param->setLabel(kParamStatHSVMeanLabel);
            param->setHint(kParamStatHSVMeanHint);
            param->setDefault(1.0, 1.0, 1.0);
			param->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
			param->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
			param->setDoubleType(eDoubleTypeScale);
			param->setIncrement(0.0001);
			param->setDigits(6);
            param->setDimensionLabels("h", "s", "v");
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }


        // analyzeFrameHSV
        {
            PushButtonParamDescriptor *param = desc.definePushButtonParam(kParamAnalyzeFrameHSV);
            param->setLabel(kParamAnalyzeFrameHSVLabel);
            param->setHint(kParamAnalyzeFrameHSVHint);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        // clearFrameHSV
        {
            PushButtonParamDescriptor *param = desc.definePushButtonParam(kParamClearFrameHSV);
            param->setLabel(kParamClearFrameHSVLabel);
            param->setHint(kParamClearFrameHSVHint);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

    }


    {
        GroupParamDescriptor* group = desc.defineGroupParam(kParamGroupLuma);
        group->setOpen(false);
        if (group) {
            group->setLabel(kParamGroupLuma);
            if (page) {
                page->addChild(*group);
            }
        }

        {
            ChoiceParamDescriptor* param = desc.defineChoiceParam(kParamLuminanceMath);
            param->setLabel(kParamLuminanceMathLabel);
            param->setHint(kParamLuminanceMathHint);
            assert(param->getNOptions() == eLuminanceMathRec709);
            param->appendOption(kParamLuminanceMathOptionRec709, kParamLuminanceMathOptionRec709Hint);
            /*assert(param->getNOptions() == eLuminanceMathRec2020);
            param->appendOption(kParamLuminanceMathOptionRec2020, kParamLuminanceMathOptionRec2020Hint);
            assert(param->getNOptions() == eLuminanceMathACESAP0);
            param->appendOption(kParamLuminanceMathOptionACESAP0, kParamLuminanceMathOptionACESAP0Hint);
            assert(param->getNOptions() == eLuminanceMathACESAP1);
            param->appendOption(kParamLuminanceMathOptionACESAP1, kParamLuminanceMathOptionACESAP1Hint);
            assert(param->getNOptions() == eLuminanceMathCcir601);
            param->appendOption(kParamLuminanceMathOptionCcir601, kParamLuminanceMathOptionCcir601Hint);*/
            assert(param->getNOptions() == eLuminanceMathAverage);
            param->appendOption(kParamLuminanceMathOptionAverage, kParamLuminanceMathOptionAverageHint);
            assert(param->getNOptions() == eLuminanceMathMaximum);
            param->appendOption(kParamLuminanceMathOptionMaximum, kParamLuminanceMathOptionMaximumHint);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }
        {
            Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamMaxLumaPix);
            param->setDoubleType(eDoubleTypeXYAbsolute);
            param->setUseHostNativeOverlayHandle(true);
            param->setLabel(kParamMaxLumaPixLabel);
            param->setHint(kParamMaxLumaPixHint);
            param->setDimensionLabels("x", "y");
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        {
            Double3DParamDescriptor* param = desc.defineDouble3DParam(kParamMaxLumaPixVal);
            param->setLabel(kParamMaxLumaPixValLabel);
            param->setDimensionLabels("r", "g", "b");
            param->setHint(kParamMaxLumaPixValHint);
            param->setDefault(0.0, 0.0, 0.0);
			param->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
			param->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
			param->setDoubleType(eDoubleTypeScale);
			param->setIncrement(0.0001);
			param->setDigits(6);
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        {
            Double2DParamDescriptor* param = desc.defineDouble2DParam(kParamMinLumaPix);
            param->setDoubleType(eDoubleTypeXYAbsolute);
            param->setUseHostNativeOverlayHandle(true);
            param->setLabel(kParamMinLumaPixLabel);
            param->setHint(kParamMinLumaPixHint);
            param->setDimensionLabels("x", "y");
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        {
            Double3DParamDescriptor* param = desc.defineDouble3DParam(kParamMinLumaPixVal);
            param->setLabel(kParamMinLumaPixValLabel);
            param->setDimensionLabels("r", "g", "b");
            param->setHint(kParamMinLumaPixValHint);
            param->setDefault(0.0, 0.0, 0.0);
			param->setRange(-DBL_MAX, -DBL_MAX, -DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX);
			param->setDisplayRange(-10000., -10000., -10000., 10000, 10000, 10000);
			param->setDoubleType(eDoubleTypeScale);
			param->setIncrement(0.0001);
			param->setDigits(6);
            param->setEvaluateOnChange(false);
            param->setAnimates(true);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        // analyzeFrameLuma
        {
            PushButtonParamDescriptor *param = desc.definePushButtonParam(kParamAnalyzeFrameLuma);
            param->setLabel(kParamAnalyzeFrameLumaLabel);
            param->setHint(kParamAnalyzeFrameLumaHint);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

        // clearFrameLuma
        {
            PushButtonParamDescriptor *param = desc.definePushButtonParam(kParamClearFrameLuma);
            param->setLabel(kParamClearFrameLumaLabel);
            param->setHint(kParamClearFrameLumaHint);
            if (group) {
                param->setParent(*group);
            }
            if (page) {
                page->addChild(*param);
            }
        }

    }
} 


ImageEffect*
BalancePluginFactory::createInstance(OfxImageEffectHandle handle, ContextEnum /*context*/)
{
    return new BalancePlugin(handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static BalancePluginFactory balancePlugin;
    p_FactoryArray.push_back(&balancePlugin);
}

