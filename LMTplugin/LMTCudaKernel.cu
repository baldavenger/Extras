const float PIE = 3.14159265358979323846264338327950288f;
const float X_BRK = 0.0078125f;
const float Y_BRK = 0.155251141552511f;
const float A = 10.5402377416545f;
const float B = 0.0729055341958355f;
//float3 RGB_2_YAB_MAT[3] = { {1.0f/3.0f, 1.0f/2.0f, 0.0f},
//								  {1.0f/3.0f, -1.0f/4.0f, 0.433012701892219f}, 
//								  {1.0f/3.0f, -1.0f/4.0f, -0.433012701892219f} };

__device__ float clamp(float in, float low, float high)
{
float out;
out = in < low ? low : in > high ? high : in;
return out;
}

__device__ float3* mult_f_f33(float f, float3 A[3])
{
for( int i = 0; i < 3; ++i )
{
A[i].x *= f;
A[i].y *= f;
A[i].z *= f;
}
return A;
}

__device__ float3 mult_f3_f33(float3 In, const float3 A[3])
{
float3 out;
out.x = In.x * A[0].x + In.y * A[0].y + In.z * A[0].z;
out.y = In.x * A[1].x + In.y * A[1].y + In.z * A[1].z;
out.z = In.x * A[2].x + In.y * A[2].y + In.z * A[2].z;
return out;
}

__device__ float3* invert_f33(float3 A[3])
{
float3 result[3];
float det =   A[0].x * A[1].y * A[2].z
			+ A[0].y * A[1].z * A[2].x
			+ A[0].z * A[1].x * A[2].y
			- A[2].x * A[1].y * A[0].z
			- A[2].y * A[1].z * A[0].x
			- A[2].z * A[1].x * A[0].y;
			
if( det != 0.0f )
{
result[0].x = A[1].y * A[2].z - A[1].z * A[2].y;
result[0].y = A[2].y * A[0].z - A[2].z * A[0].y;
result[0].z = A[0].y * A[1].z - A[0].z * A[1].y;
result[1].x = A[2].x * A[1].z - A[1].x * A[2].z;
result[1].y = A[0].x * A[2].z - A[2].x * A[0].z;
result[1].z = A[1].x * A[0].z - A[0].x * A[1].z;
result[2].x = A[1].x * A[2].y - A[2].x * A[1].y;
result[2].y = A[2].x * A[0].y - A[0].x * A[2].y;
result[2].z = A[0].x * A[1].y - A[1].x * A[0].y;

A =  mult_f_f33( 1.0f / det, result);
}
return A;
}

__device__ float3 rgb_2_yab(float3 rgb)
{
float3 RGB_2_YAB_MAT[3] = { {1.0f/3.0f, 1.0f/2.0f, 0.0f},
							{1.0f/3.0f, -1.0f/4.0f, 0.433012701892219f}, 
							{1.0f/3.0f, -1.0f/4.0f, -0.433012701892219f} };
float3 yab;
yab = mult_f3_f33(rgb, RGB_2_YAB_MAT);
return yab;
}

__device__ float3 yab_2_ych(float3 yab)
{
float3 ych;
ych = yab;
float yo = yab.y * yab.y + yab.z * yab.z;
ych.y = sqrtf(yo);
ych.z = atan2f(yab.z, yab.y) * (180.0f / PIE);
if (ych.z < 0.0f)
{
ych.z += 360.0f;
}
return ych;
}

__device__ float3 ych_2_yab(float3 ych) 
{
float3 yab;
yab.x = ych.x;
float h = ych.z * (PIE / 180.0f);
yab.y = ych.y * cosf(h);
yab.z = ych.y * sinf(h);
return yab;
}

__device__ float3 yab_2_rgb(float3 yab)
{
float3 RGB_2_YAB_MAT[3] = { {1.0f/3.0f, 1.0f/2.0f, 0.0f},
							{1.0f/3.0f, -1.0f/4.0f, 0.433012701892219f}, 
							{1.0f/3.0f, -1.0f/4.0f, -0.433012701892219f} };
float3 rgb;
float3* abc; 
abc = invert_f33(RGB_2_YAB_MAT);
rgb = mult_f3_f33(yab, abc);
return rgb;
}

__device__ float3 scale_C(float3 rgb, float percentC)
{
float3 ych, yab;
yab = rgb_2_yab(rgb);
ych = yab_2_ych(yab);
ych.y *= percentC;
yab = ych_2_yab(ych);
rgb = yab_2_rgb(yab);
return rgb;
}

__device__ float lin_to_ACEScct(float in)
{
float out;
if (in <= X_BRK){
out = A * in + B;
} else {
out = (log2f(in) + 9.72f) / 17.52f;
}
return out;
}

__device__ float ACEScct_to_lin(float in)
{
float out;    
if (in > Y_BRK){
out = powf(2.0f, in * 17.52f - 9.72f);
} else {
out = (in - B) / A;
}
return out;
}

__device__ float3 ACES_to_ACEScct(float3 in)
{
float3 out;

//AP0 to AP1
out.x =  1.4514393161f * in.x + -0.2365107469f * in.y + -0.2149285693f * in.z;
out.y = -0.0765537734f * in.x +  1.1762296998f * in.y + -0.0996759264f * in.z;
out.z =  0.0083161484f * in.x + -0.0060324498f * in.y +  0.9977163014f * in.z;

// Linear to ACEScct
out.x = lin_to_ACEScct(out.x);
out.y = lin_to_ACEScct(out.y);
out.z = lin_to_ACEScct(out.z);

return out;
}

__device__ float3 ACEScct_to_ACES(float3 in)
{
float3 lin, out;

// ACEScct to linear
lin.x = ACEScct_to_lin(in.x);
lin.y = ACEScct_to_lin(in.y);
lin.z = ACEScct_to_lin(in.z);

// AP1 to AP0
out.x =  0.6954522414f * lin.x +  0.1406786965f * lin.y +  0.1638690622f * lin.z;
out.y =  0.0447945634f * lin.x +  0.8596711185f * lin.y +  0.0955343182f * lin.z;
out.z = -0.0055258826f * lin.x +  0.0040252103f * lin.y +  1.0015006723f * lin.z;

return out;
}

__device__ float3 ASCCDL_inACEScct
(
float3 acesIn, 
float SLOPE[3],
float OFFSET[3],
float POWER[3],
float SAT
)
{

acesIn = ACES_to_ACEScct(acesIn);

//acesIn.x = powf(clamp((acesIn.x * SLOPE[0]) + OFFSET[0], 0.0f, 1.0f), POWER[0]);
//acesIn.y = powf(clamp((acesIn.y * SLOPE[1]) + OFFSET[1], 0.0f, 1.0f), POWER[1]);
//acesIn.z = powf(clamp((acesIn.z * SLOPE[2]) + OFFSET[2], 0.0f, 1.0f), POWER[2]);

float sopR = clamp((acesIn.x * SLOPE[0]) + OFFSET[0], 0.0f, 1.0f);
float sopG = clamp((acesIn.y * SLOPE[1]) + OFFSET[1], 0.0f, 1.0f);
float sopB = clamp((acesIn.z * SLOPE[2]) + OFFSET[2], 0.0f, 1.0f);

acesIn.x = powf(sopR, POWER[0]);
acesIn.y = powf(sopG, POWER[1]);
acesIn.z = powf(sopB, POWER[2]);

float luma = 0.2126f *acesIn.x + 0.7152f * acesIn.y + 0.0722f * acesIn.z;

float satClamp = clamp(SAT, 0.0f, 10.0f);    
acesIn.x = luma + satClamp * (acesIn.x - luma);
acesIn.y = luma + satClamp * (acesIn.y - luma);
acesIn.z = luma + satClamp * (acesIn.z - luma);

acesIn = ACEScct_to_ACES(acesIn);

return acesIn;
}

__device__ float3 gamma_adjust_linear(float3 rgbIn, float GAMMA, float PIVOT)
{
const float SCALAR = PIVOT / powf(PIVOT, GAMMA);

if (rgbIn.x > 0.0f){ rgbIn.x = powf(rgbIn.x, GAMMA) * SCALAR;}
if (rgbIn.y > 0.0f){ rgbIn.y = powf(rgbIn.y, GAMMA) * SCALAR;}
if (rgbIn.z > 0.0f){ rgbIn.z = powf(rgbIn.z, GAMMA) * SCALAR;}
return rgbIn;
}

__device__ float interpolate1D(float2 table[], float p, int t)
{
if( p <= table[0].x ) return table[0].y;
if( p >= table[t - 1].x ) return table[t - 1].y;

for( int i = 0; i < t - 1; ++i )
{
if( table[i].x <= p && p < table[i+1].x )
{
float s = (p - table[i].x) / (table[i+1].x - table[i].x);
return table[i].y * ( 1.0f - s ) + table[i+1].y * s;
}
}
return 0.0f;
}

__device__ float cubic_basis_shaper(float x, float w)
{
  float4 M[4] = { { -1./6,  3./6, -3./6,  1./6 },
                {  3./6, -6./6,  3./6,  0./6 },
                { -3./6,  0./6,  3./6,  0./6 },
                {  1./6,  4./6,  1./6,  0./6 } };
  
float knots[5] = { -w/2.0f, -w/4.0f, 0.0f, w/4.0f, w/2.0f };

float y = 0.0f;
if ((x > knots[0]) && (x < knots[4])) {  
float knot_coord = (x - knots[0]) * 4.0f/w;  
int j = knot_coord;
float t = knot_coord - j;

float monomials[4] = { t*t*t, t*t, t, 1. };

if ( j == 3) {
y = monomials[0] * M[0].x + monomials[1] * M[1].x + 
	monomials[2] * M[2].x + monomials[3] * M[3].x;
} else if ( j == 2) {
y = monomials[0] * M[0].y + monomials[1] * M[1].y + 
	monomials[2] * M[2].y + monomials[3] * M[3].y;
} else if ( j == 1) {
y = monomials[0] * M[0].z + monomials[1] * M[1].z + 
	monomials[2] * M[2].z + monomials[3] * M[3].z;
} else if ( j == 0) {
y = monomials[0] * M[0].w + monomials[1] * M[1].w + 
	monomials[2] * M[2].w + monomials[3] * M[3].w;
} else {
y = 0.0f;
}
}

return y * 3/2.0f;
}

__device__ float center_hue( float hue, float centerH)
{
float hueCentered = hue - centerH;
if (hueCentered < -180.0f) hueCentered = hueCentered + 360.0f;
else if (hueCentered > 180.0f) hueCentered = hueCentered - 360.0f;
return hueCentered;
}

__device__ float uncenter_hue( float hueCentered, float centerH)
{
float hue = hueCentered + centerH;
if (hue < 0.0f) hue = hue + 360.0f;
else if (hue > 360.0f) hue = hue - 360.0f;
return hue;
}

__device__ float3 rotate_H_in_H(float3 rgb, float centerH, float widthH, float degreesShift)
{
float3 ych, yab;
yab = rgb_2_yab(rgb);
ych = yab_2_ych(yab);

float centeredHue = center_hue(ych.z, centerH);
float f_H = cubic_basis_shaper(centeredHue, widthH);

float old_hue = centeredHue;
float new_hue = centeredHue + degreesShift;
float2 table[2] = {{0.0f, old_hue}, {1.0f, new_hue}};
float blended_hue = interpolate1D(table, f_H, 2);
 
if (f_H > 0.0f)
{
ych.z = uncenter_hue(blended_hue, centerH);
}

yab = ych_2_yab(ych);
rgb = yab_2_rgb(yab);
return rgb;
}

__device__ float3 scale_C_at_H
( 
float3 rgb, 
float centerH,
float widthH,
float percentC
)
{
float3 ych, yab, new_rgb;
new_rgb = rgb;
yab = rgb_2_yab(rgb);
ych = yab_2_ych(yab);
if (ych.y > 0.0f) {
float centeredHue = center_hue(ych.z, centerH);
float f_H = cubic_basis_shaper(centeredHue, widthH);
if (f_H > 0.0) {
float3 new_ych = ych;
new_ych.y = ych.y * (f_H * (percentC - 1.0f) + 1.0f);
yab = ych_2_yab(new_ych);
new_rgb = yab_2_rgb(yab);
} else { 
new_rgb = rgb; 
}
}
return new_rgb;
}

__device__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)
{
float3 Aces;
Aces.x = p_R;
Aces.y = p_G;
Aces.z = p_B;

Aces = scale_C(Aces, 0.7f);

float SLOPE[3] = {1.0f, 1.0f, 0.94f};
float OFFSET[3] = {0.0f, 0.0f, 0.02f};
float POWER[3] = {1.0f, 1.0f, 1.0f};
float SAT = 1.0f;

Aces = ASCCDL_inACEScct(Aces, SLOPE, OFFSET, POWER, SAT);

Aces = gamma_adjust_linear(Aces, 1.5f, 0.18f);

Aces = rotate_H_in_H(Aces, 0.0f, 30.0f, 5.0f);

Aces = rotate_H_in_H(Aces, 80.0f, 60.0f, -15.0f);

Aces = rotate_H_in_H(Aces, 52.0f, 50.0f, -14.0f);

Aces = scale_C_at_H(Aces, 45.0f, 40.0f, 1.4f);

Aces = rotate_H_in_H(Aces, 190.0f, 40.0f, 30.0f);

Aces = scale_C_at_H(Aces, 240.0f, 120.0f, 1.4f);

return Aces;
}

__global__ void LMTKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, float p_Scale1, float p_Scale2, 
float p_Scale3, float p_Scale4, float p_Scale5, float p_Scale6, float p_Scale7, float p_Scale8, float p_Scale9, float p_Scale10, 
float p_Scale11, float p_Scale12, float p_Scale13, float p_Scale14, float p_Scale15, float p_Scale16, float p_Scale17, float p_Scale18, 
float p_Scale19, float p_Scale20, float p_Scale21, float p_Scale22, float p_Scale23, float p_Scale24, float p_Scale25, float p_Scale26, 
float p_Scale27, float p_Scale28, float p_Scale29, float p_Scale30, float p_Scale31)
{
   const int x = blockIdx.x * blockDim.x + threadIdx.x;
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   
	
   if ((x < p_Width) && (y < p_Height))
   {
	const int index = ((y * p_Width) + x) * 4;
	
	float3 Aces;
	Aces.x = p_Input[index + 0];
	Aces.y = p_Input[index + 1];
	Aces.z = p_Input[index + 2];
	
	Aces = scale_C(Aces, p_Scale1);

	float SLOPE[3] = {p_Scale2, p_Scale3, p_Scale4};
	float OFFSET[3] = {p_Scale5, p_Scale6, p_Scale7};
	float POWER[3] = {p_Scale8, p_Scale9, p_Scale10};
	float SAT = p_Scale11;

	Aces = ASCCDL_inACEScct(Aces, SLOPE, OFFSET, POWER, SAT);

	Aces = gamma_adjust_linear(Aces, p_Scale12, p_Scale13);

	Aces = rotate_H_in_H(Aces, p_Scale14, p_Scale15, p_Scale16);

	Aces = rotate_H_in_H(Aces, p_Scale17, p_Scale18, p_Scale19);

	Aces = rotate_H_in_H(Aces, p_Scale20, p_Scale21, p_Scale22);

	Aces = scale_C_at_H(Aces, p_Scale23, p_Scale24, p_Scale25);

	Aces = rotate_H_in_H(Aces, p_Scale26, p_Scale27, p_Scale28);

	Aces = scale_C_at_H(Aces, p_Scale29, p_Scale30, p_Scale31);
																												   
	p_Output[index + 0] = Aces.x;
	p_Output[index + 1] = Aces.y;
	p_Output[index + 2] = Aces.z;
	p_Output[index + 3] = p_Input[index + 3];
   }
}

void RunCudaKernel(const float* p_Input, float* p_Output, int p_Width, int p_Height, float* p_Scale)
{
    dim3 threads(128, 1, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), p_Height, 1);

    LMTKernel<<<blocks, threads>>>(p_Input, p_Output, p_Width, p_Height, p_Scale[0], p_Scale[1], 
    p_Scale[2], p_Scale[3], p_Scale[4], p_Scale[5], p_Scale[6], p_Scale[7], p_Scale[8], p_Scale[9], p_Scale[10], 
    p_Scale[11], p_Scale[12], p_Scale[13], p_Scale[14], p_Scale[15], p_Scale[16], p_Scale[17], p_Scale[18], p_Scale[19], 
    p_Scale[20], p_Scale[21], p_Scale[22], p_Scale[23], p_Scale[24], p_Scale[25], p_Scale[26], p_Scale[27], p_Scale[28], 
    p_Scale[29], p_Scale[30]);
}
