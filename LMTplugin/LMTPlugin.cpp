#include "LMTPlugin.h"

#include <cstring>
#include <cmath>
#include <stdio.h>
using std::string;
#include <string> 
#include <fstream>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"

#ifdef __APPLE__
#define kPluginScript "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT"
#else
#define kPluginScript "/home/resolve/LUT"
#endif

#define kPluginName "LMT"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"LMT"

#define kPluginIdentifier "OpenFX.Yo.LMT"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

namespace {
    struct RGBValues {
        double r,g,b;
        RGBValues(double v) : r(v), g(v), b(v) {}
        RGBValues() : r(0), g(0), b(0) {}
    };
}

class LMT : public OFX::ImageProcessor
{
public:
    explicit LMT(OFX::ImageEffect& p_Instance);

    virtual void processImagesCUDA();
    //virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);
    
    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(float* p_Scale);

private:
    OFX::Image* _srcImg;
    float _scale[31];
};

LMT::LMT(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Scale);

void LMT::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(input, output, width, height, _scale);
}
/*
extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, const float* p_Input, float* p_Output);

void LMT::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, input, output);
}
*/
void LMT::multiThreadProcessImages(OfxRectI p_ProcWindow)
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
   			    dstPix[0] = srcPix[0];
			    dstPix[1] = srcPix[1];
			    dstPix[2] = srcPix[2];
			    dstPix[3] = srcPix[3];
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

void LMT::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void LMT::setScales(float* p_Scale)
{
   _scale[0] = p_Scale[0];
   _scale[1] = p_Scale[1];
   _scale[2] = p_Scale[2];
   _scale[3] = p_Scale[3];
   _scale[4] = p_Scale[4];
   _scale[5] = p_Scale[5];
   _scale[6] = p_Scale[6];
   _scale[7] = p_Scale[7];
   _scale[8] = p_Scale[8];
   _scale[9] = p_Scale[9];
   _scale[10] = p_Scale[10];
   _scale[11] = p_Scale[11];
   _scale[12] = p_Scale[12];
   _scale[13] = p_Scale[13];
   _scale[14] = p_Scale[14];
   _scale[15] = p_Scale[15];
   _scale[16] = p_Scale[16];
   _scale[17] = p_Scale[17];
   _scale[18] = p_Scale[18];
   _scale[19] = p_Scale[19];
   _scale[20] = p_Scale[20];   
   _scale[21] = p_Scale[21];
   _scale[22] = p_Scale[22];
   _scale[23] = p_Scale[23];
   _scale[24] = p_Scale[24];
   _scale[25] = p_Scale[25];
   _scale[26] = p_Scale[26];
   _scale[27] = p_Scale[27];
   _scale[28] = p_Scale[28];
   _scale[29] = p_Scale[29];
   _scale[30] = p_Scale[30];
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class LMTPlugin : public OFX::ImageEffect
{
public:
    explicit LMTPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);
    
     /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(LMT &p_LMT, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;
    
    OFX::DoubleParam* m_Scale1;
    OFX::DoubleParam* m_Scale2;
    OFX::DoubleParam* m_Scale3;
    OFX::DoubleParam* m_Scale4;
    OFX::DoubleParam* m_Scale5;
    OFX::DoubleParam* m_Scale6;
    OFX::DoubleParam* m_Scale7;
    OFX::DoubleParam* m_Scale8;
    OFX::DoubleParam* m_Scale9;
    OFX::DoubleParam* m_Scale10;
    OFX::DoubleParam* m_Scale11;
    OFX::DoubleParam* m_Scale12;
    OFX::DoubleParam* m_Scale13;
    OFX::DoubleParam* m_Scale14;
    OFX::DoubleParam* m_Scale15;
    OFX::DoubleParam* m_Scale16;
    OFX::DoubleParam* m_Scale17;
    OFX::DoubleParam* m_Scale18;
    OFX::DoubleParam* m_Scale19;
    OFX::DoubleParam* m_Scale20;
    OFX::DoubleParam* m_Scale21;
    OFX::DoubleParam* m_Scale22;
    OFX::DoubleParam* m_Scale23;
    OFX::DoubleParam* m_Scale24;
    OFX::DoubleParam* m_Scale25;
    OFX::DoubleParam* m_Scale26;
    OFX::DoubleParam* m_Scale27;
    OFX::DoubleParam* m_Scale28;
    OFX::DoubleParam* m_Scale29;
    OFX::DoubleParam* m_Scale30;
    OFX::DoubleParam* m_Scale31;

    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
};

LMTPlugin::LMTPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    
    m_Scale1 = fetchDoubleParam("Scale1");
    m_Scale2 = fetchDoubleParam("Scale2");
    m_Scale3 = fetchDoubleParam("Scale3");
    m_Scale4 = fetchDoubleParam("Scale4");
    m_Scale5 = fetchDoubleParam("Scale5");
    m_Scale6 = fetchDoubleParam("Scale6");
    m_Scale7 = fetchDoubleParam("Scale7");
    m_Scale8 = fetchDoubleParam("Scale8");
    m_Scale9 = fetchDoubleParam("Scale9");
    m_Scale10 = fetchDoubleParam("Scale10");
    m_Scale11 = fetchDoubleParam("Scale11");
    m_Scale12 = fetchDoubleParam("Scale12");
    m_Scale13 = fetchDoubleParam("Scale13");
    m_Scale14 = fetchDoubleParam("Scale14");
    m_Scale15 = fetchDoubleParam("Scale15");
    m_Scale16 = fetchDoubleParam("Scale16");
    m_Scale17 = fetchDoubleParam("Scale17");
    m_Scale18 = fetchDoubleParam("Scale18");
    m_Scale19 = fetchDoubleParam("Scale19");
    m_Scale20 = fetchDoubleParam("Scale20");
    m_Scale21 = fetchDoubleParam("Scale21");
    m_Scale22 = fetchDoubleParam("Scale22");
    m_Scale23 = fetchDoubleParam("Scale23");
    m_Scale24 = fetchDoubleParam("Scale24");
    m_Scale25 = fetchDoubleParam("Scale25");
    m_Scale26 = fetchDoubleParam("Scale26");
    m_Scale27 = fetchDoubleParam("Scale27");
    m_Scale28 = fetchDoubleParam("Scale28");
    m_Scale29 = fetchDoubleParam("Scale29");
    m_Scale30 = fetchDoubleParam("Scale30");
    m_Scale31 = fetchDoubleParam("Scale31");

    m_Path = fetchStringParam("path");
	m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");

}

void LMTPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        LMT LMT(*this);
        setupAndProcess(LMT, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool LMTPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
   
    
    
    if (m_SrcClip)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void LMTPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    
    if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	
	if(p_ParamName == "button1")
    {
       
	string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "// LMTPlugin DCTL export\n" \
	"\n" \
	"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
	"{\n" \
	"	float r = p_R;\n" \
	"	float g = p_G;\n" \
	"	float b = p_B;\n" \
	"\n" \
	"	return make_float3(r, g, b);\n" \
	"}\n");
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
	}	
	}
	}

	if(p_ParamName == "button2")
    {
    
	string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".nk to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".nk").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, " Group {\n" \
	" name LMT\n" \
	" selected true\n" \
	"}\n" \
	" Input {\n" \
  	" name Input1\n" \
	" }\n" \
	" Output {\n" \
  	" name Output1\n" \
	" }\n" \
	"end_group\n");
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
	}	
	}
	}
    
    
}


void LMTPlugin::setupAndProcess(LMT& p_LMT, const OFX::RenderArguments& p_Args)
{
    // Get the dst image
    std::auto_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the src image
    std::auto_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    // Check to see if the bit depth and number of components are the same
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }
    
    float _scale[31];
    
    _scale[0] = m_Scale1->getValueAtTime(p_Args.time);
    _scale[1] = m_Scale2->getValueAtTime(p_Args.time);
    _scale[2] = m_Scale3->getValueAtTime(p_Args.time);
    _scale[3] = m_Scale4->getValueAtTime(p_Args.time);
    _scale[4] = m_Scale5->getValueAtTime(p_Args.time);
    _scale[5] = m_Scale6->getValueAtTime(p_Args.time);
    _scale[6] = m_Scale7->getValueAtTime(p_Args.time);
    _scale[7] = m_Scale8->getValueAtTime(p_Args.time);
    _scale[8] = m_Scale9->getValueAtTime(p_Args.time);
    _scale[9] = m_Scale10->getValueAtTime(p_Args.time);
    _scale[10] = m_Scale11->getValueAtTime(p_Args.time);
    _scale[11] = m_Scale12->getValueAtTime(p_Args.time);
    _scale[12] = m_Scale13->getValueAtTime(p_Args.time);
    _scale[13] = m_Scale14->getValueAtTime(p_Args.time);
    _scale[14] = m_Scale15->getValueAtTime(p_Args.time);
    _scale[15] = m_Scale16->getValueAtTime(p_Args.time);
    _scale[16] = m_Scale17->getValueAtTime(p_Args.time);
    _scale[17] = m_Scale18->getValueAtTime(p_Args.time);
    _scale[18] = m_Scale19->getValueAtTime(p_Args.time);
    _scale[19] = m_Scale20->getValueAtTime(p_Args.time);
    _scale[20] = m_Scale21->getValueAtTime(p_Args.time);
    _scale[21] = m_Scale22->getValueAtTime(p_Args.time);
    _scale[22] = m_Scale23->getValueAtTime(p_Args.time);
    _scale[23] = m_Scale24->getValueAtTime(p_Args.time);
    _scale[24] = m_Scale25->getValueAtTime(p_Args.time);
    _scale[25] = m_Scale26->getValueAtTime(p_Args.time);
    _scale[26] = m_Scale27->getValueAtTime(p_Args.time);
    _scale[27] = m_Scale28->getValueAtTime(p_Args.time);
    _scale[28] = m_Scale29->getValueAtTime(p_Args.time);
    _scale[29] = m_Scale30->getValueAtTime(p_Args.time);
    _scale[30] = m_Scale31->getValueAtTime(p_Args.time);

    // Set the images
    p_LMT.setDstImg(dst.get());
    p_LMT.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_LMT.setGPURenderArgs(p_Args);

    // Set the render window
    p_LMT.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_LMT.setScales(_scale);

    // Call the base class process member, this will call the derived templated process code
    p_LMT.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

LMTPluginFactory::LMTPluginFactory()
    : OFX::PluginFactoryHelper<LMTPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void LMTPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    // Basic labels
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    // Add the supported contexts, only filter at the moment
    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);

    // Add supported pixel depths
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    // Set a few flags
    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // Setup OpenCL and CUDA render capability flags
    p_Desc.setSupportsOpenCLRender(true);
    p_Desc.setSupportsCudaRender(true);
}

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(1.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setDoubleType(eDoubleTypeScale);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}


void LMTPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip only in the filter context
    // Create the mandated source clip
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    // Make some pages and to things in
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");
    
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "Scale1", "scale_C", "scale", 0);
    param->setDefault(0.7);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale2", "slopeR", "scale", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale3", "slopeG", "scale", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale4", "slopeB", "scale", 0);
    param->setDefault(0.94);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale5", "offsetR", "scale", 0);
    param->setDefault(0.0);
    param->setRange(-10.0, 10.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-2.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale6", "offsetG", "scale", 0);
    param->setDefault(0.0);
    param->setRange(-10.0, 10.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-2.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale7", "offsetB", "scale", 0);
    param->setDefault(0.02);
    param->setRange(-10.0, 10.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-2.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale8", "powerR", "scale", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale9", "powerG", "scale", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale10", "powerB", "scale", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale11", "sat", "scale", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 3.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale12", "gamma_adjust_linear", "scale", 0);
    param->setDefault(1.5);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale13", "pivot", "scale", 0);
    param->setDefault(0.18);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale14", "rotate_H_in_Hue1", "scale", 0);
    param->setDefault(0.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale15", "range1", "scale", 0);
    param->setDefault(30.0);
    param->setRange(0.0, 180.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 180.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale16", "shift1", "scale", 0);
    param->setDefault(5.0);
    param->setRange(-90.0, 90.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-90.0, 90.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale17", "rotate_H_in_Hue2", "scale", 0);
    param->setDefault(80.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale18", "range2", "scale", 0);
    param->setDefault(60.0);
    param->setRange(0.0, 180.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 180.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale19", "shift2", "scale", 0);
    param->setDefault(-15.0);
    param->setRange(-90.0, 90.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-90.0, 90.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale20", "rotate_H_in_H3", "scale", 0);
    param->setDefault(52.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale21", "range3", "scale", 0);
    param->setDefault(50.0);
    param->setRange(0.0, 180.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 180.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale22", "shift3", "scale", 0);
    param->setDefault(-14.0);
    param->setRange(-90.0, 90.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-90.0, 90.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale23", "scale_C_at_Hue1", "scale", 0);
    param->setDefault(45.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale24", "rangeC1", "scale", 0);
    param->setDefault(40.0);
    param->setRange(0.0, 180.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 180.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale25", "scaleC1", "scale", 0);
    param->setDefault(1.4);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale26", "rotate_H_in_Hue4", "scale", 0);
    param->setDefault(190.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale27", "range4", "scale", 0);
    param->setDefault(40.0);
    param->setRange(0.0, 180.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 180.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale28", "shift4", "scale", 0);
    param->setDefault(30.0);
    param->setRange(-90.0, 90.0);
    param->setIncrement(0.01);
    param->setDisplayRange(-90.0, 90.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale29", "scale_C_at_Hue2", "scale", 0);
    param->setDefault(240.0);
    param->setRange(0.0, 360.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 360.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale30", "rangeC2", "scale", 0);
    param->setDefault(120.0);
    param->setRange(0.0, 180.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 180.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "Scale31", "scaleC2", "scale", 0);
    param->setDefault(1.4);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 3.0);
    page->addChild(*param);

    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("info");
    param->setLabel("Info");
    page->addChild(*param);
    }
    
    {    
    GroupParamDescriptor* script = p_Desc.defineGroupParam("Script Export");
    script->setOpen(false);
    script->setHint("export DCTL and Nuke script");
      if (page) {
            page->addChild(*script);
            }
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("button1");
    param->setLabel("Export DCTL");
    param->setHint("create DCTL version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("button2");
    param->setLabel("Export Nuke script");
    param->setHint("create NUKE version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
	StringParamDescriptor* param = p_Desc.defineStringParam("name");
	param->setLabel("Name");
	param->setHint("overwrites if the same");
	param->setDefault("LMT");
	param->setParent(*script);
	page->addChild(*param);
	}
	{
	StringParamDescriptor* param = p_Desc.defineStringParam("path");
	param->setLabel("Directory");
	param->setHint("make sure it's the absolute path");
	param->setStringType(eStringTypeFilePath);
	param->setDefault(kPluginScript);
	param->setFilePathExists(false);
	param->setParent(*script);
	page->addChild(*param);
	}
	}        
    
}

ImageEffect* LMTPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new LMTPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static LMTPluginFactory LMTPlugin;
    p_FactoryArray.push_back(&LMTPlugin);
}
