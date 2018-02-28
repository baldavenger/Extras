#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif
#include <map>
#include <stdio.h>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const char *KernelSource = "\n" \
"__constant float PIE = 3.14159265358979323846264338327950288f; \n" \
"__constant float X_BRK = 0.0078125f; \n" \
"__constant float Y_BRK = 0.155251141552511f; \n" \
"__constant float A = 10.5402377416545f; \n" \
"__constant float B = 0.0729055341958355f; \n" \
"float3* mult_f_f33(float f, float3 A[3]); \n" \
"float3 mult_f3_f33(float3 In, const float3 A[3]); \n" \
"float3 rgb_2_yab(float3 rgb); \n" \
"float3 yab_2_ych(float3 yab); \n" \
"float3* invert_f33(float3 A[3]); \n" \
"float3 yab_2_rgb(float3 yab); \n" \
"float3 ych_2_yab(float3 ych);  \n" \
"float ACEScct_to_lin(float in); \n" \
"float lin_to_ACEScct(float in); \n" \
"float3 scale_C(float3 rgb, float percentC); \n" \
"float3 scale_C_at_H \n" \
"(  \n" \
"float3 rgb,  \n" \
"float centerH, \n" \
"float widthH, \n" \
"float percentC \n" \
"); \n" \
"float3 ACES_to_ACEScct(float3 in); \n" \
"float3 ACEScct_to_ACES(float3 in); \n" \
"float3 gamma_adjust_linear(float3 rgbIn, float GAMMA, float PIVOT); \n" \
"float interpolate1D(float2 table[], float p, int t); \n" \
"float cubic_basis_shaper(float x, float w); \n" \
"float center_hue( float hue, float centerH); \n" \
"float uncenter_hue( float hueCentered, float centerH); \n" \
"float3 rotate_H_in_H(float3 rgb, float centerH, float widthH, float degreesShift); \n" \
"float3 ASCCDL_inACEScct \n" \
"( \n" \
"float3 acesIn,  \n" \
"float SLOPE[3], \n" \
"float OFFSET[3], \n" \
"float POWER[3], \n" \
"float SAT \n" \
"); \n" \
"float3* mult_f_f33(float f, float3 A[3]) \n" \
"{ \n" \
"for( int i = 0; i < 3; ++i ) \n" \
"{ \n" \
"A[i].x *= f; \n" \
"A[i].y *= f; \n" \
"A[i].z *= f; \n" \
"} \n" \
"return A; \n" \
"} \n" \
"float3 mult_f3_f33(float3 In, const float3 A[3]) \n" \
"{ \n" \
"float3 out; \n" \
"out.x = In.x * A[0].x + In.y * A[0].y + In.z * A[0].z; \n" \
"out.y = In.x * A[1].x + In.y * A[1].y + In.z * A[1].z; \n" \
"out.z = In.x * A[2].x + In.y * A[2].y + In.z * A[2].z; \n" \
"return out; \n" \
"} \n" \
"float3* invert_f33(float3 A[3]) \n" \
"{ \n" \
"float3 result[3]; \n" \
"float det =   A[0].x * A[1].y * A[2].z \n" \
"			+ A[0].y * A[1].z * A[2].x \n" \
"			+ A[0].z * A[1].x * A[2].y \n" \
"			- A[2].x * A[1].y * A[0].z \n" \
"			- A[2].y * A[1].z * A[0].x \n" \
"			- A[2].z * A[1].x * A[0].y;		 \n" \
"if( det != 0.0f ) \n" \
"{ \n" \
"result[0].x = A[1].y * A[2].z - A[1].z * A[2].y; \n" \
"result[0].y = A[2].y * A[0].z - A[2].z * A[0].y; \n" \
"result[0].z = A[0].y * A[1].z - A[0].z * A[1].y; \n" \
"result[1].x = A[2].x * A[1].z - A[1].x * A[2].z; \n" \
"result[1].y = A[0].x * A[2].z - A[2].x * A[0].z; \n" \
"result[1].z = A[1].x * A[0].z - A[0].x * A[1].z; \n" \
"result[2].x = A[1].x * A[2].y - A[2].x * A[1].y; \n" \
"result[2].y = A[2].x * A[0].y - A[0].x * A[2].y; \n" \
"result[2].z = A[0].x * A[1].y - A[1].x * A[0].y; \n" \
" \n" \
"A =  mult_f_f33( 1.0f / det, result); \n" \
"} \n" \
"return A; \n" \
"} \n" \
"float3 rgb_2_yab(float3 rgb) \n" \
"{ \n" \
"float3 RGB_2_YAB_MAT[3] = { {1.0f/3.0f, 1.0f/2.0f, 0.0f}, \n" \
"							{1.0f/3.0f, -1.0f/4.0f, 0.433012701892219f},  \n" \
"							{1.0f/3.0f, -1.0f/4.0f, -0.433012701892219f} }; \n" \
"float3 yab; \n" \
"yab = mult_f3_f33(rgb, RGB_2_YAB_MAT); \n" \
"return yab; \n" \
"} \n" \
"float3 yab_2_ych(float3 yab) \n" \
"{ \n" \
"float3 ych; \n" \
"ych = yab; \n" \
"float yo = yab.y * yab.y + yab.z * yab.z; \n" \
"ych.y = sqrt(yo); \n" \
"ych.z = atan2(yab.z, yab.y) * (180.0f / PIE); \n" \
"if (ych.z < 0.0f) \n" \
"{ \n" \
"ych.z += 360.0f; \n" \
"} \n" \
"return ych; \n" \
"} \n" \
"float3 ych_2_yab(float3 ych)  \n" \
"{ \n" \
"float3 yab; \n" \
"yab.x = ych.x; \n" \
"float h = ych.z * (PIE / 180.0f); \n" \
"yab.y = ych.y * cos(h); \n" \
"yab.z = ych.y * sin(h); \n" \
"return yab; \n" \
"} \n" \
"float3 yab_2_rgb(float3 yab) \n" \
"{ \n" \
"float3 RGB_2_YAB_MAT[3] = { {1.0f/3.0f, 1.0f/2.0f, 0.0f}, \n" \
"							{1.0f/3.0f, -1.0f/4.0f, 0.433012701892219f},  \n" \
"							{1.0f/3.0f, -1.0f/4.0f, -0.433012701892219f} }; \n" \
"float3 rgb; \n" \
"float3* abc;  \n" \
"abc = invert_f33(RGB_2_YAB_MAT); \n" \
"rgb = mult_f3_f33(yab, abc); \n" \
"return rgb; \n" \
"} \n" \
"float3 scale_C(float3 rgb, float percentC) \n" \
"{ \n" \
"float3 ych, yab; \n" \
"yab = rgb_2_yab(rgb); \n" \
"ych = yab_2_ych(yab); \n" \
"ych.y *= percentC; \n" \
"yab = ych_2_yab(ych); \n" \
"rgb = yab_2_rgb(yab); \n" \
"return rgb; \n" \
"} \n" \
"float lin_to_ACEScct(float in) \n" \
"{ \n" \
"float out; \n" \
"if (in <= X_BRK){ \n" \
"out = A * in + B; \n" \
"} else { \n" \
"out = (log2(in) + 9.72f) / 17.52f; \n" \
"} \n" \
"return out; \n" \
"} \n" \
"float ACEScct_to_lin(float in) \n" \
"{ \n" \
"float out;     \n" \
"if (in > Y_BRK){ \n" \
"out = pow(2.0f, in * 17.52f - 9.72f); \n" \
"} else { \n" \
"out = (in - B) / A; \n" \
"} \n" \
"return out; \n" \
"} \n" \
"float3 ACES_to_ACEScct(float3 in) \n" \
"{ \n" \
"float3 out; \n" \
"out.x =  1.4514393161f * in.x + -0.2365107469f * in.y + -0.2149285693f * in.z; \n" \
"out.y = -0.0765537734f * in.x +  1.1762296998f * in.y + -0.0996759264f * in.z; \n" \
"out.z =  0.0083161484f * in.x + -0.0060324498f * in.y +  0.9977163014f * in.z; \n" \
"out.x = lin_to_ACEScct(out.x); \n" \
"out.y = lin_to_ACEScct(out.y); \n" \
"out.z = lin_to_ACEScct(out.z); \n" \
"return out; \n" \
"} \n" \
"float3 ACEScct_to_ACES(float3 in) \n" \
"{ \n" \
"float3 lin, out; \n" \
"lin.x = ACEScct_to_lin(in.x); \n" \
"lin.y = ACEScct_to_lin(in.y); \n" \
"lin.z = ACEScct_to_lin(in.z); \n" \
"out.x =  0.6954522414f * lin.x +  0.1406786965f * lin.y +  0.1638690622f * lin.z; \n" \
"out.y =  0.0447945634f * lin.x +  0.8596711185f * lin.y +  0.0955343182f * lin.z; \n" \
"out.z = -0.0055258826f * lin.x +  0.0040252103f * lin.y +  1.0015006723f * lin.z; \n" \
"return out; \n" \
"} \n" \
"float3 ASCCDL_inACEScct \n" \
"( \n" \
"float3 acesIn,  \n" \
"float SLOPE[3], \n" \
"float OFFSET[3], \n" \
"float POWER[3], \n" \
"float SAT \n" \
") \n" \
"{ \n" \
"acesIn = ACES_to_ACEScct(acesIn); \n" \
"float sopR = clamp((acesIn.x * SLOPE[0]) + OFFSET[0], 0.0f, 1.0f); \n" \
"float sopG = clamp((acesIn.y * SLOPE[1]) + OFFSET[1], 0.0f, 1.0f); \n" \
"float sopB = clamp((acesIn.z * SLOPE[2]) + OFFSET[2], 0.0f, 1.0f); \n" \
"acesIn.x = pow(sopR, POWER[0]); \n" \
"acesIn.y = pow(sopG, POWER[1]); \n" \
"acesIn.z = pow(sopB, POWER[2]); \n" \
"float luma = 0.2126f *acesIn.x + 0.7152f * acesIn.y + 0.0722f * acesIn.z; \n" \
"float satClamp = clamp(SAT, 0.0f, 10.0f);     \n" \
"acesIn.x = luma + satClamp * (acesIn.x - luma); \n" \
"acesIn.y = luma + satClamp * (acesIn.y - luma); \n" \
"acesIn.z = luma + satClamp * (acesIn.z - luma); \n" \
"acesIn = ACEScct_to_ACES(acesIn); \n" \
"return acesIn; \n" \
"} \n" \
"float3 gamma_adjust_linear(float3 rgbIn, float GAMMA, float PIVOT) \n" \
"{ \n" \
"const float SCALAR = PIVOT / pow(PIVOT, GAMMA); \n" \
"if (rgbIn.x > 0.0f){ rgbIn.x = pow(rgbIn.x, GAMMA) * SCALAR;} \n" \
"if (rgbIn.y > 0.0f){ rgbIn.y = pow(rgbIn.y, GAMMA) * SCALAR;} \n" \
"if (rgbIn.z > 0.0f){ rgbIn.z = pow(rgbIn.z, GAMMA) * SCALAR;} \n" \
"return rgbIn; \n" \
"} \n" \
"float interpolate1D(float2 table[], float p, int t) \n" \
"{ \n" \
"if( p <= table[0].x ) return table[0].y; \n" \
"if( p >= table[t - 1].x ) return table[t - 1].y; \n" \
"for( int i = 0; i < t - 1; ++i ) \n" \
"{ \n" \
"if( table[i].x <= p && p < table[i+1].x ) \n" \
"{ \n" \
"float s = (p - table[i].x) / (table[i+1].x - table[i].x); \n" \
"return table[i].y * ( 1.0f - s ) + table[i+1].y * s; \n" \
"} \n" \
"} \n" \
"return 0.0f; \n" \
"} \n" \
"float cubic_basis_shaper(float x, float w) \n" \
"{ \n" \
"  float4 M[4] = { { -1./6,  3./6, -3./6,  1./6 }, \n" \
"                {  3./6, -6./6,  3./6,  0./6 }, \n" \
"                { -3./6,  0./6,  3./6,  0./6 }, \n" \
"                {  1./6,  4./6,  1./6,  0./6 } };  \n" \
"float knots[5] = { -w/2.0f, -w/4.0f, 0.0f, w/4.0f, w/2.0f }; \n" \
"float y = 0.0f; \n" \
"if ((x > knots[0]) && (x < knots[4])) {   \n" \
"float knot_coord = (x - knots[0]) * 4.0f/w;   \n" \
"int j = knot_coord; \n" \
"float t = knot_coord - j; \n" \
"float monomials[4] = { t*t*t, t*t, t, 1. }; \n" \
"if ( j == 3) { \n" \
"y = monomials[0] * M[0].x + monomials[1] * M[1].x +  \n" \
"	monomials[2] * M[2].x + monomials[3] * M[3].x; \n" \
"} else if ( j == 2) { \n" \
"y = monomials[0] * M[0].y + monomials[1] * M[1].y +  \n" \
"	monomials[2] * M[2].y + monomials[3] * M[3].y; \n" \
"} else if ( j == 1) { \n" \
"y = monomials[0] * M[0].z + monomials[1] * M[1].z +  \n" \
"	monomials[2] * M[2].z + monomials[3] * M[3].z; \n" \
"} else if ( j == 0) { \n" \
"y = monomials[0] * M[0].w + monomials[1] * M[1].w +  \n" \
"	monomials[2] * M[2].w + monomials[3] * M[3].w; \n" \
"} else { \n" \
"y = 0.0f; \n" \
"} \n" \
"} \n" \
"return y * 3/2.0f; \n" \
"} \n" \
"float center_hue( float hue, float centerH) \n" \
"{ \n" \
"float hueCentered = hue - centerH; \n" \
"if (hueCentered < -180.0f) hueCentered = hueCentered + 360.0f; \n" \
"else if (hueCentered > 180.0f) hueCentered = hueCentered - 360.0f; \n" \
"return hueCentered; \n" \
"} \n" \
"float uncenter_hue( float hueCentered, float centerH) \n" \
"{ \n" \
"float hue = hueCentered + centerH; \n" \
"if (hue < 0.0f) hue = hue + 360.0f; \n" \
"else if (hue > 360.0f) hue = hue - 360.0f; \n" \
"return hue; \n" \
"} \n" \
"float3 rotate_H_in_H(float3 rgb, float centerH, float widthH, float degreesShift) \n" \
"{ \n" \
"float3 ych, yab; \n" \
"yab = rgb_2_yab(rgb); \n" \
"ych = yab_2_ych(yab); \n" \
"float centeredHue = center_hue(ych.z, centerH); \n" \
"float f_H = cubic_basis_shaper(centeredHue, widthH); \n" \
"float old_hue = centeredHue; \n" \
"float new_hue = centeredHue + degreesShift; \n" \
"float2 table[2] = {{0.0f, old_hue}, {1.0f, new_hue}}; \n" \
"float blended_hue = interpolate1D(table, f_H, 2); \n" \
"if (f_H > 0.0f) \n" \
"{ \n" \
"ych.z = uncenter_hue(blended_hue, centerH); \n" \
"} \n" \
"yab = ych_2_yab(ych); \n" \
"rgb = yab_2_rgb(yab); \n" \
"return rgb; \n" \
"} \n" \
"float3 scale_C_at_H \n" \
"(  \n" \
"float3 rgb,  \n" \
"float centerH, \n" \
"float widthH, \n" \
"float percentC \n" \
") \n" \
"{ \n" \
"float3 ych, yab, new_rgb; \n" \
"new_rgb = rgb; \n" \
"yab = rgb_2_yab(rgb); \n" \
"ych = yab_2_ych(yab); \n" \
"if (ych.y > 0.0f) { \n" \
"float centeredHue = center_hue(ych.z, centerH); \n" \
"float f_H = cubic_basis_shaper(centeredHue, widthH); \n" \
"if (f_H > 0.0) { \n" \
"float3 new_ych = ych; \n" \
"new_ych.y = ych.y * (f_H * (percentC - 1.0f) + 1.0f); \n" \
"yab = ych_2_yab(new_ych); \n" \
"new_rgb = yab_2_rgb(yab); \n" \
"} else {  \n" \
"new_rgb = rgb;  \n" \
"} \n" \
"} \n" \
"return new_rgb; \n" \
"} \n" \
"__kernel void LMTKernel(__global const float* p_Input, __global float* p_Output, int p_Width, int p_Height, int p_ACESin, int p_ACESout,  \n" \
"float p_Scale1, float p_Scale2, float p_Scale3, float p_Scale4, float p_Scale5, float p_Scale6, float p_Scale7, float p_Scale8,  \n" \
"float p_Scale9, float p_Scale10, float p_Scale11, float p_Scale12, float p_Scale13, float p_Scale14, float p_Scale15, float p_Scale16,  \n" \
"float p_Scale17, float p_Scale18, float p_Scale19, float p_Scale20, float p_Scale21, float p_Scale22, float p_Scale23, float p_Scale24,  \n" \
"float p_Scale25, float p_Scale26, float p_Scale27, float p_Scale28, float p_Scale29, float p_Scale30, float p_Scale31) \n" \
"{ \n" \
"const int x = get_global_id(0); \n" \
"const int y = get_global_id(1); \n" \
"if ((x < p_Width) && (y < p_Height)) \n" \
"{   \n" \
"const int index = (y * p_Width + x) * 4; \n" \
"float3 Aces; \n" \
"Aces.x = p_Input[index + 0]; \n" \
"Aces.y = p_Input[index + 1]; \n" \
"Aces.z = p_Input[index + 2]; \n" \
"switch (p_ACESin) \n" \
"{ \n" \
"case 0: \n" \
"{ \n" \
"Aces = ACEScct_to_ACES(Aces); \n" \
"} \n" \
"break; \n" \
"case 1: \n" \
"{ \n" \
"} \n" \
"break; \n" \
"default:  \n" \
"Aces = ACEScct_to_ACES(Aces); \n" \
"} \n" \
"Aces = scale_C(Aces, p_Scale1); \n" \
"float SLOPE[3] = {p_Scale2, p_Scale3, p_Scale4}; \n" \
"float OFFSET[3] = {p_Scale5, p_Scale6, p_Scale7}; \n" \
"float POWER[3] = {p_Scale8, p_Scale9, p_Scale10}; \n" \
"float SAT = p_Scale11; \n" \
"Aces = ASCCDL_inACEScct(Aces, SLOPE, OFFSET, POWER, SAT); \n" \
"Aces = gamma_adjust_linear(Aces, p_Scale12, p_Scale13); \n" \
"Aces = rotate_H_in_H(Aces, p_Scale14, p_Scale15, p_Scale16); \n" \
"Aces = rotate_H_in_H(Aces, p_Scale17, p_Scale18, p_Scale19); \n" \
"Aces = rotate_H_in_H(Aces, p_Scale20, p_Scale21, p_Scale22); \n" \
"Aces = scale_C_at_H(Aces, p_Scale23, p_Scale24, p_Scale25); \n" \
"Aces = rotate_H_in_H(Aces, p_Scale26, p_Scale27, p_Scale28); \n" \
"Aces = scale_C_at_H(Aces, p_Scale29, p_Scale30, p_Scale31); \n" \
"switch (p_ACESout) \n" \
"{ \n" \
"case 0: \n" \
"{ \n" \
"Aces = ACES_to_ACEScct(Aces); \n" \
"} \n" \
"break; \n" \
" \n" \
"case 1: \n" \
"{ \n" \
"} \n" \
"break; \n" \
"default:  \n" \
"Aces = ACES_to_ACEScct(Aces); \n" \
"} \n" \
"p_Output[index + 0] = Aces.x; \n" \
"p_Output[index + 1] = Aces.y; \n" \
"p_Output[index + 2] = Aces.z; \n" \
"p_Output[index + 3] = p_Input[index + 3]; \n" \
"} \n" \
"} \n" \
"\n";

class Locker
{
public:
	Locker()
	{
#ifdef _WIN64
		InitializeCriticalSection(&mutex);
#else
		pthread_mutex_init(&mutex, NULL);
#endif
	}

	~Locker()
	{
#ifdef _WIN64
		DeleteCriticalSection(&mutex);
#else
		pthread_mutex_destroy(&mutex);
#endif
	}

	void Lock()
	{
#ifdef _WIN64
		EnterCriticalSection(&mutex);
#else
		pthread_mutex_lock(&mutex);
#endif
	}

	void Unlock()
	{
#ifdef _WIN64
		LeaveCriticalSection(&mutex);
#else
		pthread_mutex_unlock(&mutex);
#endif
	}

private:
#ifdef _WIN64
	CRITICAL_SECTION mutex;
#else
	pthread_mutex_t mutex;
#endif
};


void CheckError(cl_int p_Error, const char* p_Msg)
{
	if (p_Error != CL_SUCCESS)
	{
		fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
	}
}

void RunOpenCLKernel(void* p_CmdQ, const float* p_Input, float* p_Output, int p_Width, int p_Height, int p_ACESin, int p_ACESout, float* p_Scale)
{
	cl_int error;

	cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

	// store device id and kernel per command queue (required for multi-GPU systems)
	static std::map<cl_command_queue, cl_device_id> deviceIdMap;
	static std::map<cl_command_queue, cl_kernel> kernelMap;

	static Locker locker; // simple lock to control access to the above maps from multiple threads

	locker.Lock();

	// find the device id corresponding to the command queue
	cl_device_id deviceId = NULL;
	if (deviceIdMap.find(cmdQ) == deviceIdMap.end())
	{
		error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
		CheckError(error, "Unable to get the device");

		deviceIdMap[cmdQ] = deviceId;
	}
	else
	{
		deviceId = deviceIdMap[cmdQ];
	}


	cl_kernel kernel = NULL;
	
	cl_context clContext = NULL;
	error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
	CheckError(error, "Unable to get the context");

	cl_program program = clCreateProgramWithSource(clContext, 1, (const char **)&KernelSource, NULL, &error);
	CheckError(error, "Unable to create program");

	error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CheckError(error, "Unable to build program");

	kernel = clCreateKernel(program, "LMTKernel", &error);
	CheckError(error, "Unable to create kernel");

		
	locker.Unlock();

	int count = 0;
	error = clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Input);
	error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &p_Output);
	error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
	error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
	error |= clSetKernelArg(kernel, count++, sizeof(int), &p_ACESin);
	error |= clSetKernelArg(kernel, count++, sizeof(int), &p_ACESout);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[0]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[1]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[2]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[3]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[4]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[5]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[6]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[7]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[8]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[9]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[10]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[11]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[12]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[13]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[14]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[15]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[16]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[17]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[18]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[19]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[20]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[21]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[22]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[23]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[24]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[25]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[26]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[27]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[28]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[29]);
	error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Scale[30]);
	CheckError(error, "Unable to set kernel arguments");

	size_t localWorkSize[2], globalWorkSize[2];
	clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), localWorkSize, NULL);
	localWorkSize[1] = 1;
	globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
	globalWorkSize[1] = p_Height;

	clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
}
