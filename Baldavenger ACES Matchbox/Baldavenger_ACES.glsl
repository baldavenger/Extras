// Baldavenger ACES Matchbox Shader

#version 120
uniform sampler2D front;
uniform float adsk_result_w, adsk_result_h;
uniform float p_Exposure;
uniform int p_IDT;
uniform int p_LMT;
uniform int p_RRT;
uniform int p_ODT;

struct Chromaticities {
vec2 red,green,blue,white;
};

struct SplineMapPoint {
float x,y;
};
    
struct float5 {
float x,y,z,w,m;
};

struct float6 {
float a,b,c,d,e,f;
};

struct float10 {
float a,b,c,d,e,f,g,h,i,j;
};

struct SegmentedSplineParams_c5 {
float6 coefsLow,coefsHigh;
SplineMapPoint minPoint,midPoint,maxPoint;
float slopeLow,slopeHigh;
};
    
struct SegmentedSplineParams_c9 {
float10 coefsLow,coefsHigh;
SplineMapPoint minPoint,midPoint,maxPoint;
float slopeLow,slopeHigh;
};

struct TsPoint {
float x; float y; float slope;
};

struct TsParams {
TsPoint Min; TsPoint Mid; TsPoint Max;
float6 coefsLow; float6 coefsHigh;
};

#define REF_PT				((7120.0 - 1520.0) / 8000.0 * (100.0 / 55.0) - log10(0.18)) * 1.0
#define AP0_2_XYZ_MAT		RGBtoXYZ(AP0)
#define XYZ_2_AP0_MAT		XYZtoRGB(AP0)
#define AP1_2_XYZ_MAT		RGBtoXYZ(AP1)
#define XYZ_2_AP1_MAT		XYZtoRGB(AP1)
#define AP0_2_AP1_MAT		XYZ_2_AP1_MAT * AP0_2_XYZ_MAT
#define AP1_2_AP0_MAT		XYZ_2_AP0_MAT * AP1_2_XYZ_MAT
#define AP1_RGB2Y			vec3(AP1_2_XYZ_MAT[0][1], AP1_2_XYZ_MAT[1][1], AP1_2_XYZ_MAT[2][1])
#define ODT_SAT_MAT			calc_sat_adjust_matrix( ODT_SAT_FACTOR, AP1_RGB2Y)
#define D60_2_D65_CAT		calculate_cat_matrix( AP0.white, REC709_PRI.white)
#define RRT_SAT_MAT			calc_sat_adjust_matrix( RRT_SAT_FACTOR, AP1_RGB2Y)
#define CINEMA_WHITE		48.0
#define CINEMA_BLACK		pow(10.0, log10(0.02))

mat3 MM = mat3( vec3(0.5, -1.0, 0.5), vec3(-1.0, 1.0, 0.5), vec3(0.5, 0.0, 0.0) );
const float TINY = 1e-10;
const float DIM_SURROUND_GAMMA = 0.9811;
const float ODT_SAT_FACTOR = 0.93;
const float MIN_STOP_SDR = -6.5;
const float MAX_STOP_SDR = 6.5;
const float MIN_STOP_RRT = -15.0;
const float MAX_STOP_RRT = 18.0;
const float MIN_LUM_SDR = 0.02;
const float MAX_LUM_SDR = 48.0;
const float MIN_LUM_RRT = 0.0001;
const float MAX_LUM_RRT = 10000.0;
const float RRT_GLOW_GAIN = 0.05;
const float RRT_GLOW_MID = 0.08;
const float RRT_RED_SCALE = 0.82;
const float RRT_RED_PIVOT = 0.03;
const float RRT_RED_HUE = 0.0;
const float RRT_RED_WIDTH = 135.0;
const float RRT_SAT_FACTOR = 0.96;
const float X_BRK = 0.0078125;
const float Y_BRK = 0.155251141552511;
const float A = 10.5402377416545;
const float B = 0.0729055341958355;
const float sqrt3over4 = 0.433012701892219;
const float pq_m1 = 0.1593017578125;
const float pq_m2 = 78.84375;
const float pq_c1 = 0.8359375;
const float pq_c2 = 18.8515625;
const float pq_c3 = 18.6875;
const float pq_C = 10000.0;

const mat3 CDD_TO_CID = mat3(
vec3(0.75573, 0.05901, 0.16134),
vec3(0.22197, 0.96928, 0.07406), 
vec3(0.02230, -0.02829, 0.76460)
);

const mat3 EXP_TO_ACES = mat3(
vec3(0.72286, 0.11923, 0.01427),
vec3(0.12630, 0.76418, 0.08213),
vec3(0.15084, 0.11659, 0.90359)
);

const Chromaticities AP0 = Chromaticities(
vec2(0.7347, 0.2653),vec2(0.0, 1.0),
vec2(0.0001, -0.077),vec2(0.32168, 0.33767)
);

const Chromaticities AP1 = Chromaticities(
vec2(0.713, 0.293),vec2(0.165, 0.83),
vec2(0.128, 0.044),vec2(0.32168, 0.33767)
);

const Chromaticities REC709_PRI = Chromaticities(
vec2(0.64, 0.33),vec2(0.3, 0.6),
vec2(0.15, 0.06),vec2(0.3127, 0.329)
);

const Chromaticities P3D60_PRI = Chromaticities(
vec2(0.68, 0.32),vec2(0.265, 0.69),
vec2(0.15, 0.06),vec2(0.32168, 0.33767)
);
const Chromaticities P3D65_PRI = Chromaticities(
vec2(0.68, 0.32),vec2(0.265, 0.69),
vec2(0.15, 0.06),vec2(0.3127, 0.329)
);

const Chromaticities P3DCI_PRI = Chromaticities(
vec2(0.68, 0.32),vec2(0.265, 0.69),
vec2(0.15, 0.06),vec2(0.314, 0.351)
);

const Chromaticities ARRI_ALEXA_WG_PRI = Chromaticities(
vec2(0.684, 0.313),vec2(0.221, 0.848),
vec2(0.0861, -0.102),vec2(0.3127, 0.329)
);

const Chromaticities REC2020_PRI = Chromaticities(
vec2(0.708, 0.292),vec2(0.17, 0.797),
vec2(0.131, 0.046),vec2(0.3127, 0.329)
);

const Chromaticities RIMMROMM_PRI = Chromaticities(
vec2(0.7347, 0.2653),vec2(0.1596, 0.8404),
vec2(0.0366, 0.0001),vec2(0.3457, 0.3585)
);

const mat3 CONE_RESP_MAT_BRADFORD = mat3(
vec3(0.8951, -0.7502, 0.0389),
vec3(0.2664, 1.7135, -0.0685),
vec3(-0.1614, 0.0367, 1.0296)
);

const mat3 CONE_RESP_MAT_CAT02 = mat3(
vec3(0.7328, -0.7036, 0.003),
vec3(0.4296, 1.6975, 0.0136),
vec3(-0.1624, 0.0061, 0.9834)
);

const mat3 AP1_2_AP0_MAT_B = mat3(
vec3(0.6954522414, 0.0447945634, -0.0055258826), 
vec3(0.1406786965, 0.8596711185, 0.0040252103), 
vec3(0.1638690622, 0.0955343182, 1.0015006723) );


float data6[6];
float data10[10];

float getData6(int id) {
for (int i=0; i<6; i++) {
if (i == id) return data6[i];
}}

float getData10(int id) {
for (int i=0; i<10; i++) {
if (i == id) return data10[i];
}}

float min_f3( vec3 a)
{
return min( a.x, min( a.y, a.z));
}

float max_f3( vec3 a)
{
return max( a.x, max( a.y, a.z));
}

float log10( float x) {
return (1.0 / log(10.0)) * log(x);
}

float clip( float v)
{
return min(v, 1.0);
}

vec3 clip_f3( vec3 ya)
{
vec3 Out;
Out.x = clip( ya.x); Out.y = clip( ya.y); Out.z = clip( ya.z);
return Out;
}

vec3 pow_f3( vec3 a, float b)
{
vec3 Out;
Out.x = pow(a.x, b); Out.y = pow(a.y, b); Out.z = pow(a.z, b);
return Out;
}

float pow10( float x)
{
return pow(10.0, x);
}

vec3 pow10_f3( vec3 a)
{
vec3 Out;
Out.x = pow10(a.x); Out.y = pow10(a.y); Out.z = pow10(a.z);
return Out;
}

vec3 log10_f3( vec3 a)
{
vec3 Out;
Out.x = log10(a.x); Out.y = log10(a.y); Out.z = log10(a.z);
return Out;
}

float _sign( float x)
{
float y;
if (x < 0.0) y = -1.0;
else if (x > 0.0) y = 1.0;
else y = 0.0;
return y;
}

vec3 mult_f3_f33( vec3 X, mat3 A)
{
float r[3];
float x[3] = float[3](X.x, X.y, X.z);
for( int i = 0; i < 3; ++i){
r[i] = 0.0f;
for( int j = 0; j < 3; ++j){
r[i] = r[i] + x[j] * A[j][i];}}
return vec3(r[0], r[1], r[2]);
}

mat3 invert_f33( mat3 A) {
mat3 R;
mat3 result;
mat3 a = mat3(vec3(A[0][0], A[0][1], A[0][2]),
vec3(A[1][0], A[1][1], A[1][2]),
vec3(A[2][0], A[2][1], A[2][2]));
float det =   a[0][0] * a[1][1] * a[2][2]
+ a[0][1] * a[1][2] * a[2][0]
+ a[0][2] * a[1][0] * a[2][1]
- a[2][0] * a[1][1] * a[0][2]
- a[2][1] * a[1][2] * a[0][0]
- a[2][2] * a[1][0] * a[0][1];
if( det != 0.0 )
{
result[0][0] = a[1][1] * a[2][2] - a[1][2] * a[2][1];
result[0][1] = a[2][1] * a[0][2] - a[2][2] * a[0][1];
result[0][2] = a[0][1] * a[1][2] - a[0][2] * a[1][1];
result[1][0] = a[2][0] * a[1][2] - a[1][0] * a[2][2];
result[1][1] = a[0][0] * a[2][2] - a[2][0] * a[0][2];
result[1][2] = a[1][0] * a[0][2] - a[0][0] * a[1][2];
result[2][0] = a[1][0] * a[2][1] - a[2][0] * a[1][1];
result[2][1] = a[2][0] * a[0][1] - a[0][0] * a[2][1];
result[2][2] = a[0][0] * a[1][1] - a[1][0] * a[0][1];
R = mat3(vec3(result[0][0], result[0][1], result[0][2]), 
vec3(result[1][0], result[1][1], result[1][2]), vec3(result[2][0], result[2][1], result[2][2]));
return (1.0 / det) * R;
}
R = mat3(vec3(1.0, 0.0, 0.0), 
vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0));
return R;
}

float interpolate1D( mat2 table, float p) {
if( p <= table[0][0] ) return table[0][1];
if( p >= table[1][0] ) return table[1][1];
if( table[0][0] <= p && p < table[1][0] ){
float s = (p - table[0][0]) / (table[1][0] - table[0][0]);
return table[0][1] * ( 1.0 - s ) + table[1][1] * s;
}
return 0.0;
}

float interpolate1D11( float tableIN[11], float tableOUT[11], float p) {
if( p <= tableIN[0] ) return tableOUT[0];
if( p >= tableIN[10] ) return tableOUT[10];
for( int i = 0; i < 10; ++i ){
if( tableIN[i] <= p && p < tableIN[i+1] )
{
float s = (p - tableIN[i]) / (tableIN[i+1] - tableIN[i]);
return tableOUT[i] * ( 1.0 - s ) + tableOUT[i+1] * s;
}}
return 0.0f;
}

mat3 RGBtoXYZ( Chromaticities N) {
mat3 M = mat3(vec3(N.red.x, N.red.y, 1.0 - (N.red.x + N.red.y)),
vec3(N.green.x, N.green.y, 1.0 - (N.green.x + N.green.y)),
vec3(N.blue.x, N.blue.y, 1.0 - (N.blue.x + N.blue.y)));
vec3 wh = vec3(N.white.x / N.white.y, 1.0, (1.0 - (N.white.x + N.white.y)) / N.white.y);
wh = invert_f33(M) * wh;
mat3 WH = mat3(vec3(wh.x, 0.0, 0.0), 
vec3(0.0, wh.y, 0.0), vec3(0.0, 0.0, wh.z));
M = M * WH;
return M;
}

mat3 XYZtoRGB( Chromaticities N)
{
mat3 M = invert_f33(RGBtoXYZ(N));
return M;
}

float SLog3_to_linear( float SLog )
{
float Out = 0.0;
if (SLog >= 171.2102946929 / 1023.0){
Out = pow(10.0, (SLog * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01;
} else {
Out = (SLog * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0);}
return Out;
}

float vLogToLinScene( float x)
{
const float cutInv = 0.181;
const float b = 0.00873;
const float c = 0.241514;
const float d = 0.598206;
if (x <= cutInv)
return (x - 0.125) / 5.6;
else
return pow(10.0, (x - d) / c) - b;
}

float SLog1_to_lin( float SLog, float b, float ab, float w)
{
float lin = 0.0;
if (SLog >= ab)
lin = ( pow(10.0, ( ( ( SLog - b) / ( w - b) - 0.616596 - 0.03) / 0.432699)) - 0.037584) * 0.9;
else if (SLog < ab)
lin = ( ( ( SLog - b) / ( w - b) - 0.030001222851889303) / 5.0) * 0.9;
return lin;
}

float SLog2_to_lin( float SLog, float b, float ab, float w)
{
float lin = 0.0;
if (SLog >= ab)
lin = ( 219.0 * ( pow(10.0, ( ( ( SLog - b) / ( w - b) - 0.616596 - 0.03) / 0.432699)) - 0.037584) / 155.0) * 0.9;
else if (SLog < ab)
lin = ( ( ( SLog - b) / ( w - b) - 0.030001222851889303) / 3.53881278538813) * 0.9;
return lin;
}

float CanonLog_to_linear ( float clog)
{
float Out = 0.0;
if(clog < 0.12512248)
Out = -( pow( 10.0, ( 0.12512248 - clog ) / 0.45310179 ) - 1.0 ) / 10.1596;
else
Out = ( pow( 10.0, ( clog - 0.12512248 ) / 0.45310179 ) - 1.0 ) / 10.1596;
return Out;
}

float CanonLog2_to_linear ( float clog2)
{
float Out = 0.0;
if(clog2 < 0.092864125)
Out = -( pow( 10.0, ( 0.092864125 - clog2 ) / 0.24136077 ) - 1.0 ) / 87.099375;
else
Out = ( pow( 10.0, ( clog2 - 0.092864125 ) / 0.24136077 ) - 1.0 ) / 87.099375;
return Out;
}

float CanonLog3_to_linear ( float clog3)
{
float Out = 0.0;
if(clog3 < 0.097465473)
Out = -( pow( 10.0, ( 0.12783901 - clog3 ) / 0.36726845 ) - 1.0 ) / 14.98325;
else if(clog3 <= 0.15277891)
Out = ( clog3 - 0.12512219 ) / 1.9754798;
else
Out = ( pow( 10.0, ( clog3 - 0.12240537 ) / 0.36726845 ) - 1.0 ) / 14.98325;
return Out;
}

float Log3G10_to_linear_2016 ( float log3g10)
{
float a, b, c, mirror, linear;
a = 0.224282;
b = 155.975327;
c = 0.01;
mirror = 1.0;
if (log3g10 < 0.0){
mirror = -1.0;
log3g10 = -log3g10;}
linear = (pow(10.0, log3g10 / a) - 1.0) / b;
linear = linear * mirror - c;
return linear;
}

float Log3G10_to_linear ( float log3g10)
{
float a, b, c, g, linear;
a = 0.224282; b = 155.975327; c = 0.01; g = 15.1927;
linear = log3g10 < 0.0 ? (log3g10 / g) : (pow(10.0, log3g10 / a) - 1.0) / b;
linear = linear - c;
return linear;
}

vec3 XYZ_2_xyY( vec3 XYZ)
{
vec3 xyY;
float divisor = (XYZ.x + XYZ.y + XYZ.z);
if (divisor == 0.0) divisor = 1e-10;
xyY.x = XYZ.x / divisor;
xyY.y = XYZ.y / divisor;
xyY.z = XYZ.y;
return xyY;
}

vec3 xyY_2_XYZ( vec3 xyY)
{
vec3 XYZ;
XYZ.x = xyY.x * xyY.z / max( xyY.y, 1e-10);
XYZ.y = xyY.z;
XYZ.z = (1.0 - xyY.x - xyY.y) * xyY.z / max( xyY.y, 1e-10);
return XYZ;
}

float rgb_2_hue( vec3 rgb)
{
float hue = 0.0;
if (rgb.x == rgb.y && rgb.y == rgb.z) {
hue = 0.0;
} else {
hue = (180.0/3.14159265358979323846264338327950288) * atan( sqrt(3.0) * (rgb.y - rgb.z), 2.0 * rgb.x - rgb.y - rgb.z);
}
if (hue < 0.0) hue = hue + 360.0;
return hue;
}

float rgb_2_yc( vec3 rgb)
{
float ycRadiusWeight = 1.75;
float r = rgb.x;
float g = rgb.y;
float b = rgb.z;
float chroma = sqrt(b * (b - g) + g * (g - r) + r * (r - b));
return ( b + g + r + ycRadiusWeight * chroma) / 3.0;
}

mat3 calculate_cat_matrix( vec2 src_xy, vec2 des_xy) {
mat3 coneRespMat = CONE_RESP_MAT_BRADFORD;
vec3 src_xyY = vec3( src_xy.x, src_xy.y, 1.0 );
vec3 des_xyY = vec3( des_xy.x, des_xy.y, 1.0 );
vec3 src_XYZ = xyY_2_XYZ( src_xyY );
vec3 des_XYZ = xyY_2_XYZ( des_xyY );
vec3 src_coneResp = coneRespMat * src_XYZ;
vec3 des_coneResp = coneRespMat * des_XYZ;
mat3 vkMat = mat3(
vec3( des_coneResp.x / src_coneResp.x, 0.0, 0.0 ),
vec3( 0.0, des_coneResp.y / src_coneResp.y, 0.0 ),
vec3( 0.0, 0.0, des_coneResp.z / src_coneResp.z ) );
mat3 cat_matrix = (vkMat * invert_f33(coneRespMat)) * coneRespMat;
return cat_matrix;
}

mat3 calc_sat_adjust_matrix( float sat, vec3 rgb2Y) {
mat3 M;
M[0][0] = (1.0 - sat) * rgb2Y.x + sat;
M[1][0] = (1.0 - sat) * rgb2Y.x;
M[2][0] = (1.0 - sat) * rgb2Y.x;
M[0][1] = (1.0 - sat) * rgb2Y.y;
M[1][1] = (1.0 - sat) * rgb2Y.y + sat;
M[2][1] = (1.0 - sat) * rgb2Y.y;
M[0][2] = (1.0 - sat) * rgb2Y.z;
M[1][2] = (1.0 - sat) * rgb2Y.z;
M[2][2] = (1.0 - sat) * rgb2Y.z + sat;
mat3 R = mat3(vec3(M[0][0], M[0][1], M[0][2]), 
vec3(M[1][0], M[1][1], M[1][2]), vec3(M[2][0], M[2][1], M[2][2]));
R = transpose(R);    
return R;
}

float moncurve_f( float x, float gamma, float offs )
{
float y;
float fs = (( gamma - 1.0) / offs) * pow( offs * gamma / ( ( gamma - 1.0) * ( 1.0 + offs)), gamma);
float xb = offs / ( gamma - 1.0);
if ( x >= xb)
y = pow( ( x + offs) / ( 1.0 + offs), gamma);
else
y = x * fs;
return y;
}

float moncurve_r( float y, float gamma, float offs )
{
float x;
float yb = pow( offs * gamma / ( ( gamma - 1.0) * ( 1.0 + offs)), gamma);
float rs = pow( ( gamma - 1.0) / offs, gamma - 1.0) * pow( ( 1.0 + offs) / gamma, gamma);
if ( y >= yb)
x = ( 1.0 + offs) * pow( y, 1.0 / gamma) - offs;
else
x = y * rs;
return x;
}

vec3 moncurve_f_f3( vec3 x, float gamma, float offs)
{
vec3 y;
y.x = moncurve_f( x.x, gamma, offs); y.y = moncurve_f( x.y, gamma, offs); y.z = moncurve_f( x.z, gamma, offs);
return y;
}

vec3 moncurve_r_f3( vec3 y, float gamma, float offs)
{
vec3 x;
x.x = moncurve_r( y.x, gamma, offs); x.y = moncurve_r( y.y, gamma, offs); x.z = moncurve_r( y.z, gamma, offs);
return x;
}

float bt1886_f( float V, float gamma, float Lw, float Lb)
{
float a = pow( pow( Lw, 1.0/gamma) - pow( Lb, 1.0/gamma), gamma);
float b = pow( Lb, 1.0/gamma) / ( pow( Lw, 1.0/gamma) - pow( Lb, 1.0/gamma));
float L = a * pow( max( V + b, 0.0), gamma);
return L;
}

float bt1886_r( float L, float gamma, float Lw, float Lb)
{
float a = pow( pow( Lw, 1.0/gamma) - pow( Lb, 1.0/gamma), gamma);
float b = pow( Lb, 1.0/gamma) / ( pow( Lw, 1.0/gamma) - pow( Lb, 1.0/gamma));
float V = pow( max( L / a, 0.0), 1.0/gamma) - b;
return V;
}

vec3 bt1886_f_f3( vec3 V, float gamma, float Lw, float Lb)
{
vec3 L;
L.x = bt1886_f( V.x, gamma, Lw, Lb); L.y = bt1886_f( V.y, gamma, Lw, Lb); L.z = bt1886_f( V.z, gamma, Lw, Lb);
return L;
}

vec3 bt1886_r_f3( vec3 L, float gamma, float Lw, float Lb)
{
vec3 V;
V.x = bt1886_r( L.x, gamma, Lw, Lb); V.y = bt1886_r( L.y, gamma, Lw, Lb); V.z = bt1886_r( L.z, gamma, Lw, Lb);
return V;
}

float smpteRange_to_fullRange( float ya)
{
const float REFBLACK = ( 64.0 / 1023.0);
const float REFWHITE = ( 940.0 / 1023.0);
return (( ya - REFBLACK) / ( REFWHITE - REFBLACK));
}

float fullRange_to_smpteRange( float ya)
{
const float REFBLACK = ( 64.0 / 1023.0);
const float REFWHITE = ( 940.0 / 1023.0);
return ( ya * ( REFWHITE - REFBLACK) + REFBLACK );
}

vec3 smpteRange_to_fullRange_f3( vec3 rgbIn)
{
vec3 rgbOut;
rgbOut.x = smpteRange_to_fullRange( rgbIn.x); rgbOut.y = smpteRange_to_fullRange( rgbIn.y); rgbOut.z = smpteRange_to_fullRange( rgbIn.z);
return rgbOut;
}

vec3 fullRange_to_smpteRange_f3( vec3 rgbIn)
{
vec3 rgbOut;
rgbOut.x = fullRange_to_smpteRange( rgbIn.x); rgbOut.y = fullRange_to_smpteRange( rgbIn.y); rgbOut.z = fullRange_to_smpteRange( rgbIn.z);
return rgbOut;
}

vec3 dcdm_decode( vec3 XYZp)
{
vec3 XYZ;
XYZ.x = (52.37/48.0) * pow( XYZp.x, 2.6);
XYZ.y = (52.37/48.0) * pow( XYZp.y, 2.6);
XYZ.z = (52.37/48.0) * pow( XYZp.z, 2.6);
return XYZ;
}

vec3 dcdm_encode( vec3 XYZ)
{
vec3 XYZp;
XYZp.x = pow( (48.0/52.37) * XYZ.x, 1.0/2.6);
XYZp.y = pow( (48.0/52.37) * XYZ.y, 1.0/2.6);
XYZp.z = pow( (48.0/52.37) * XYZ.z, 1.0/2.6);
return XYZp;
}

float ST2084_2_Y( float N )
{
float Np = pow( N, 1.0 / pq_m2 );
float L = Np - pq_c1;
if ( L < 0.0 )
L = 0.0;
L = L / ( pq_c2 - pq_c3 * Np );
L = pow( L, 1.0 / pq_m1 );
return L * pq_C;
}

float Y_2_ST2084( float C )
{
float L = C / pq_C;
float Lm = pow( L, pq_m1 );
float N = ( pq_c1 + pq_c2 * Lm ) / ( 1.0 + pq_c3 * Lm );
N = pow( N, pq_m2 );
return N;
}

vec3 Y_2_ST2084_f3( vec3 ya)
{
vec3 Out;
Out.x = Y_2_ST2084( ya.x); Out.y = Y_2_ST2084( ya.y); Out.z = Y_2_ST2084( ya.z);
return Out;
}

vec3 ST2084_2_Y_f3( vec3 ya)
{
vec3 Out;
Out.x = ST2084_2_Y( ya.x); Out.y = ST2084_2_Y( ya.y); Out.z = ST2084_2_Y( ya.z);
return Out;
}

vec3 ST2084_2_HLG_1000nits_f3( vec3 PQ)
{
vec3 displayLinear = ST2084_2_Y_f3( PQ);
float Y_d = 0.2627 * displayLinear.x + 0.6780 * displayLinear.y + 0.0593 * displayLinear.z;
const float L_w = 1000.0;
const float L_b = 0.0;
const float alpha = (L_w - L_b);
const float beta = L_b;
const float gamma = 1.2;
vec3 sceneLinear;
if (Y_d == 0.0) {
sceneLinear.x = 0.0; sceneLinear.y = 0.0; sceneLinear.z = 0.0;
} else {
sceneLinear.x = pow( (Y_d - beta) / alpha, (1.0 - gamma) / gamma) * ((displayLinear.x - beta) / alpha);
sceneLinear.y = pow( (Y_d - beta) / alpha, (1.0 - gamma) / gamma) * ((displayLinear.y - beta) / alpha);
sceneLinear.z = pow( (Y_d - beta) / alpha, (1.0 - gamma) / gamma) * ((displayLinear.z - beta) / alpha);
}
const float a = 0.17883277;
const float b = 0.28466892;
const float c = 0.55991073;
vec3 HLG;
if (sceneLinear.x <= 1.0 / 12.0) {
HLG.x = sqrt(3.0 * sceneLinear.x);
} else {
HLG.x = a * log(12.0 * sceneLinear.x-b)+c;
}
if (sceneLinear.y <= 1.0 / 12.0) {
HLG.y = sqrt(3.0 * sceneLinear.y);
} else {
HLG.y = a * log(12.0 * sceneLinear.y-b)+c;
}
if (sceneLinear.z <= 1.0 / 12.0) {
HLG.z = sqrt(3.0 * sceneLinear.z);
} else {
HLG.z = a * log(12.0 * sceneLinear.z - b) + c;
}
return HLG;
}

vec3 HLG_2_ST2084_1000nits_f3( vec3 HLG)
{
const float a = 0.17883277;
const float b = 0.28466892;
const float c = 0.55991073;
const float L_w = 1000.0;
const float L_b = 0.0;
const float alpha = (L_w - L_b);
const float beta = L_b;
const float gamma = 1.2;
vec3 sceneLinear;
if ( HLG.x >= 0.0 && HLG.x <= 0.5) {
sceneLinear.x = pow(HLG.x, 2.0) / 3.0;
} else {
sceneLinear.x = (exp((HLG.x - c) / a) + b) / 12.0;
}
if ( HLG.y >= 0.0 && HLG.y <= 0.5) {
sceneLinear.y = pow(HLG.y, 2.0) / 3.0;
} else {
sceneLinear.y = (exp((HLG.y - c) / a) + b) / 12.0;
}
if ( HLG.z >= 0.0 && HLG.z <= 0.5) {
sceneLinear.z = pow(HLG.z, 2.0) / 3.0;
} else {
sceneLinear.z = (exp((HLG.z - c) / a) + b) / 12.0;
}
float Y_s = 0.2627 * sceneLinear.x + 0.6780 * sceneLinear.y + 0.0593 * sceneLinear.z;
vec3 displayLinear;
displayLinear.x = alpha * pow( Y_s, gamma - 1.0) * sceneLinear.x + beta;
displayLinear.y = alpha * pow( Y_s, gamma - 1.0) * sceneLinear.y + beta;
displayLinear.z = alpha * pow( Y_s, gamma - 1.0) * sceneLinear.z + beta;
vec3 PQ = Y_2_ST2084_f3( displayLinear);
return PQ;
}
float rgb_2_saturation( vec3 rgb)
{
return ( max( max_f3(rgb), TINY) - max( min_f3(rgb), TINY)) / max( max_f3(rgb), 1e-2);
}

SegmentedSplineParams_c5 RRT_PARAMS() {
SegmentedSplineParams_c5 A = SegmentedSplineParams_c5(float6( -4.0, -4.0, -3.1573765773, -0.4852499958, 1.8477324706, 1.8477324706), 
float6( -0.7185482425, 2.0810307172, 3.6681241237, 4.0, 4.0, 4.0), SplineMapPoint(0.18 * pow(2.0, -15.0), 0.0001), 
SplineMapPoint(0.18, 4.8), SplineMapPoint(0.18 * pow(2.0, 18.0), 10000.0), 0.0, 0.0);
return A;
}

float segmented_spline_c5_fwd( float x) {
SegmentedSplineParams_c5 C = RRT_PARAMS();
const int N_KNOTS_LOW = 4;
const int N_KNOTS_HIGH = 4;
float X = max(x, 0.0);
float logx = log10(X);
float coefsLow[6];
coefsLow[0] = C.coefsLow.a;coefsLow[1] = C.coefsLow.b;coefsLow[2] = C.coefsLow.c;
coefsLow[3] = C.coefsLow.d;coefsLow[4] = C.coefsLow.e;coefsLow[5] = C.coefsLow.f;
float coefsHigh[6];
coefsHigh[0] = C.coefsHigh.a;coefsHigh[1] = C.coefsHigh.b;coefsHigh[2] = C.coefsHigh.c;
coefsHigh[3] = C.coefsHigh.d;coefsHigh[4] = C.coefsHigh.e;coefsHigh[5] = C.coefsHigh.f;
float logy;
if ( logx <= log10(C.minPoint.x) ) { 
logy = logx * C.slopeLow + (log10(C.minPoint.y) - C.slopeLow * log10(C.minPoint.x) );
} else if (( logx > log10(C.minPoint.x) ) && ( logx < log10(C.midPoint.x) )) {
float knot_coord = float(N_KNOTS_LOW - 1) * (logx - log10(C.minPoint.x))/(log10(C.midPoint.x) - log10(C.minPoint.x));
int j = int(knot_coord);
float t = knot_coord - float(j);
vec3 cf;
data6[0] = coefsLow[0];data6[1] = coefsLow[1];data6[2] = coefsLow[2];
data6[3] = coefsLow[3];data6[4] = coefsLow[4];data6[5] = coefsLow[5];
cf.x = getData6(j); cf.y = getData6(j + 1); cf.z = getData6(j + 2);
vec3 monomials = vec3( t * t, t, 1.0 );
logy = dot( monomials, cf * transpose(MM));
} else if (( logx >= log10(C.midPoint.x) ) && ( logx < log10(C.maxPoint.x) )) {
float knot_coord = float(N_KNOTS_HIGH-1) * (logx-log10(C.midPoint.x))/(log10(C.maxPoint.x) - log10(C.midPoint.x));
int j = int(knot_coord);
float t = knot_coord - float(j);
vec3 cf;
data6[0] = coefsHigh[0];data6[1] = coefsHigh[1];data6[2] = coefsHigh[2];
data6[3] = coefsHigh[3];data6[4] = coefsHigh[4];data6[5] = coefsHigh[5];
cf.x = getData6(j); cf.y = getData6(j + 1); cf.z = getData6(j + 2); 
vec3 monomials = vec3(t * t, t, 1.0);
logy = dot( monomials, cf * transpose(MM));
} else {
logy = logx * C.slopeHigh + ( log10(C.maxPoint.y) - C.slopeHigh * log10(C.maxPoint.x) );
}
return pow(10.0, logy);
}

SegmentedSplineParams_c9 ODT_48nits() {
SegmentedSplineParams_c9 A =
SegmentedSplineParams_c9(float10( -1.6989700043, -1.6989700043, -1.4779, -1.2291, -0.8648, -0.448, 0.00518, 0.4511080334, 0.9113744414, 0.9113744414),
float10( 0.5154386965, 0.8470437783, 1.1358, 1.3802, 1.5197, 1.5985, 1.6467, 1.6746091357, 1.6878733390, 1.6878733390 ),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 * pow(2.0, -6.5) ),  0.02),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 ), 4.8),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 * pow(2.0, 6.5) ), 48.0), 0.0, 0.04);
return A;
}

SegmentedSplineParams_c9 ODT_1000nits() {
SegmentedSplineParams_c9 A =
SegmentedSplineParams_c9(float10( -4.9706219331, -3.0293780669, -2.1262, -1.5105, -1.0578, -0.4668, 0.11938, 0.7088134201, 1.2911865799, 1.2911865799 ),
float10( 0.8089132070, 1.1910867930, 1.5683, 1.9483, 2.3083, 2.6384, 2.8595, 2.9872608805, 3.0127391195, 3.0127391195 ),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 * pow(2.0, -12.0) ), 0.0001),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 ), 10.0),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 * pow(2.0, 10.0) ), 1000.0), 3.0, 0.06);
return A;
}

SegmentedSplineParams_c9 ODT_2000nits() {
SegmentedSplineParams_c9 A =
SegmentedSplineParams_c9(float10( -4.9706219331, -3.0293780669, -2.1262, -1.5105, -1.0578, -0.4668, 0.11938, 0.7088134201, 1.2911865799, 1.2911865799 ),
float10( 0.8019952042, 1.1980047958, 1.5943, 1.9973, 2.3783, 2.7684, 3.0515, 3.2746293562, 3.3274306351, 3.3274306351 ),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 * pow(2.0, -12.0) ), 0.0001),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 ), 10.0),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 * pow(2.0, 11.0) ), 2000.0), 3.0, 0.12);
return A;
}

SegmentedSplineParams_c9 ODT_4000nits() {
SegmentedSplineParams_c9 A =
SegmentedSplineParams_c9(float10( -4.9706219331, -3.0293780669, -2.1262, -1.5105, -1.0578, -0.4668, 0.11938, 0.7088134201, 1.2911865799, 1.2911865799 ),
float10( 0.7973186613, 1.2026813387, 1.6093, 2.0108, 2.4148, 2.8179, 3.1725, 3.5344995451, 3.6696204376, 3.6696204376 ),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 * pow(2.0, -12.0) ), 0.0001), 
SplineMapPoint(segmented_spline_c5_fwd( 0.18 ), 10.0),
SplineMapPoint(segmented_spline_c5_fwd( 0.18 * pow(2.0, 12.0) ), 4000.0), 3.0, 0.3);
return A;
}

float segmented_spline_c5_rev( float y) {  
SegmentedSplineParams_c5 C = RRT_PARAMS();
const int N_KNOTS_LOW = 4;
const int N_KNOTS_HIGH = 4;
float coefsLow[6];
coefsLow[0] = C.coefsLow.a;coefsLow[1] = C.coefsLow.b;coefsLow[2] = C.coefsLow.c;
coefsLow[3] = C.coefsLow.d;coefsLow[4] = C.coefsLow.e;coefsLow[5] = C.coefsLow.f;
float coefsHigh[6];
coefsHigh[0] = C.coefsHigh.a;coefsHigh[1] = C.coefsHigh.b;coefsHigh[2] = C.coefsHigh.c;
coefsHigh[3] = C.coefsHigh.d;coefsHigh[4] = C.coefsHigh.e;coefsHigh[5] = C.coefsHigh.f;
float KNOT_INC_LOW = (log10(C.midPoint.x) - log10(C.minPoint.x)) / float(N_KNOTS_LOW - 1);
float KNOT_INC_HIGH = (log10(C.maxPoint.x) - log10(C.midPoint.x)) / float(N_KNOTS_HIGH - 1);
float KNOT_Y_LOW[ N_KNOTS_LOW];
for (int i = 0; i < N_KNOTS_LOW; i += 1) {
KNOT_Y_LOW[ i] = ( coefsLow[i] + coefsLow[i+1]) / 2.0;};
float KNOT_Y_HIGH[ N_KNOTS_HIGH];
for (int i = 0; i < N_KNOTS_HIGH; i += 1) {
KNOT_Y_HIGH[ i] = ( coefsHigh[i] + coefsHigh[i+1]) / 2.0;};
float logy = log10( max(y,TINY));
float logx;
if (logy <= log10(C.minPoint.y)) {
logx = log10(C.minPoint.x);
} else if ( (logy > log10(C.minPoint.y)) && (logy <= log10(C.midPoint.y)) ) {
int j;
vec3 cf;
if ( logy > KNOT_Y_LOW[ 0] && logy <= KNOT_Y_LOW[ 1]) {
cf.x = coefsLow[0];  cf.y = coefsLow[1];  cf.z = coefsLow[2];  j = 0;
} else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) {
cf.x = coefsLow[1];  cf[ 1] = coefsLow[2];  cf.z = coefsLow[3];  j = 1;
} else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) {
cf.x = coefsLow[2];  cf.y = coefsLow[3];  cf.z = coefsLow[4];  j = 2;
} 
vec3 tmp = MM * cf;
float a = tmp.x;
float b = tmp.y;
float c = tmp.z;
c = c - logy;
float d = sqrt( b * b - 4. * a * c);
float t = ( 2.0 * c) / ( -d - b);
logx = log10(C.minPoint.x) + ( t + float(j)) * KNOT_INC_LOW;
} else if ( (logy > log10(C.midPoint.y)) && (logy < log10(C.maxPoint.y)) ) {
int j;
vec3 cf;
if ( logy > KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) {
cf.x = coefsHigh[0];  cf.y = coefsHigh[1];  cf.z = coefsHigh[2];  j = 0;
} else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) {
cf.x = coefsHigh[1];  cf.y = coefsHigh[2];  cf.z = coefsHigh[3];  j = 1;
} else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) {
cf.x = coefsHigh[2];  cf.y = coefsHigh[3];  cf.z = coefsHigh[4];  j = 2;
} 
vec3 tmp = MM * cf;
float a = tmp.x;
float b = tmp.y;
float c = tmp.z;
c = c - logy;
float d = sqrt( b * b - 4. * a * c);
float t = ( 2. * c) / ( -d - b);
logx = log10(C.midPoint.x) + ( t + float(j)) * KNOT_INC_HIGH;
} else {
logx = log10(C.maxPoint.x);
}
return pow(10.0, logx);
}

float segmented_spline_c9_fwd( float x, SegmentedSplineParams_c9 C) {    
const int N_KNOTS_LOW = 8;
const int N_KNOTS_HIGH = 8;
float logx = log10( max(x, 0.0 ));
float coefsLow[10];
coefsLow[0] = C.coefsLow.a;coefsLow[1] = C.coefsLow.b;coefsLow[2] = C.coefsLow.c;coefsLow[3] = C.coefsLow.d;
coefsLow[4] = C.coefsLow.e;coefsLow[5] = C.coefsLow.f;coefsLow[6] = C.coefsLow.g;
coefsLow[7] = C.coefsLow.h;coefsLow[8] = C.coefsLow.i;coefsLow[9] = C.coefsLow.j;
float coefsHigh[10];
coefsHigh[0] = C.coefsHigh.a;coefsHigh[1] = C.coefsHigh.b;coefsHigh[2] = C.coefsHigh.c;coefsHigh[3] = C.coefsHigh.d;
coefsHigh[4] = C.coefsHigh.e;coefsHigh[5] = C.coefsHigh.f;coefsHigh[6] = C.coefsHigh.g;
coefsHigh[7] = C.coefsHigh.h;coefsHigh[8] = C.coefsHigh.i;coefsHigh[9] = C.coefsHigh.j;
float logy;
if ( logx <= log10(C.minPoint.x) ) { 
logy = logx * C.slopeLow + ( log10(C.minPoint.y) - C.slopeLow * log10(C.minPoint.x) );
} else if (( logx > log10(C.minPoint.x) ) && ( logx < log10(C.midPoint.x) )) {
float knot_coord = float(N_KNOTS_LOW - 1) * (logx - log10(C.minPoint.x)) / (log10(C.midPoint.x) - log10(C.minPoint.x));
int j = int(knot_coord);
float t = knot_coord - float(j);
vec3 cf;
data10[0] = coefsLow[0];data10[1] = coefsLow[1];data10[2] = coefsLow[2];data10[3] = coefsLow[3];
data10[4] = coefsLow[4];data10[5] = coefsLow[5];data10[6] = coefsLow[6];
data10[7] = coefsLow[7];data10[8] = coefsLow[8];data10[9] = coefsLow[9];
cf.x = getData10(j); cf.y = getData10(j + 1); cf.z = getData10(j + 2);
vec3 monomials = vec3( t * t, t, 1.0 );
logy = dot( monomials, MM * cf);
} else if (( logx >= log10(C.midPoint.x) ) && ( logx < log10(C.maxPoint.x) )) {
float knot_coord = float(N_KNOTS_HIGH - 1) * (logx - log10(C.midPoint.x)) / (log10(C.maxPoint.x) - log10(C.midPoint.x));
int j = int(knot_coord);
float t = knot_coord - float(j);
vec3 cf;
data10[0] = coefsHigh[0];data10[1] = coefsHigh[1];data10[2] = coefsHigh[2];data10[3] = coefsHigh[3];
data10[4] = coefsHigh[4];data10[5] = coefsHigh[5];data10[6] = coefsHigh[6];
data10[7] = coefsHigh[7];data10[8] = coefsHigh[8];data10[9] = coefsHigh[9];
cf.x = getData10(j); cf.y = getData10(j + 1); cf.z = getData10(j + 2); 
vec3 monomials = vec3( t * t, t, 1.0 );
logy = dot( monomials, MM * cf);
} else {
logy = logx * C.slopeHigh + ( log10(C.maxPoint.y) - C.slopeHigh * log10(C.maxPoint.x) );
}
return pow(10.0, logy);
}

float segmented_spline_c9_rev( float y, SegmentedSplineParams_c9 C) {  
//SegmentedSplineParams_c9 C = ODT_48nits();
const int N_KNOTS_LOW = 8;
const int N_KNOTS_HIGH = 8;
float KNOT_INC_LOW = (log10(C.midPoint.x) - log10(C.minPoint.x)) / float(N_KNOTS_LOW - 1);
float KNOT_INC_HIGH = (log10(C.maxPoint.x) - log10(C.midPoint.x)) / float(N_KNOTS_HIGH - 1);
float coefsLow[10];
coefsLow[0] = C.coefsLow.a;coefsLow[1] = C.coefsLow.b;coefsLow[2] = C.coefsLow.c;coefsLow[3] = C.coefsLow.d;
coefsLow[4] = C.coefsLow.e;coefsLow[5] = C.coefsLow.f;coefsLow[6] = C.coefsLow.g;
coefsLow[7] = C.coefsLow.h;coefsLow[8] = C.coefsLow.i;coefsLow[9] = C.coefsLow.j;
float coefsHigh[10];
coefsHigh[0] = C.coefsHigh.a;coefsHigh[1] = C.coefsHigh.b;coefsHigh[2] = C.coefsHigh.c;coefsHigh[3] = C.coefsHigh.d;
coefsHigh[4] = C.coefsHigh.e;coefsHigh[5] = C.coefsHigh.f;coefsHigh[6] = C.coefsHigh.g;
coefsHigh[7] = C.coefsHigh.h;coefsHigh[8] = C.coefsHigh.i;coefsHigh[9] = C.coefsHigh.j;   
float KNOT_Y_LOW[ N_KNOTS_LOW];
for (int i = 0; i < N_KNOTS_LOW; i += 1) {
KNOT_Y_LOW[ i] = ( coefsLow[i] + coefsLow[i+1]) / 2.0;
};
float KNOT_Y_HIGH[ N_KNOTS_HIGH];
for (int i = 0; i < N_KNOTS_HIGH; i += 1) {
KNOT_Y_HIGH[ i] = ( coefsHigh[i] + coefsHigh[i+1]) / 2.0;
};
float logy = log10( max( y, TINY));
float logx;
if (logy <= log10(C.minPoint.y)) {
logx = log10(C.minPoint.x);
} else if ( (logy > log10(C.minPoint.y)) && (logy <= log10(C.midPoint.y)) ) {
int j;
vec3 cf;
if ( logy > KNOT_Y_LOW[ 0] && logy <= KNOT_Y_LOW[ 1]) {
cf.x = coefsLow[0];  cf.y = coefsLow[1];  cf.z = coefsLow[2];  j = 0;
} else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) {
cf.x = coefsLow[1];  cf.y = coefsLow[2];  cf.z = coefsLow[3];  j = 1;
} else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) {
cf.x = coefsLow[2];  cf.y = coefsLow[3];  cf.z = coefsLow[4];  j = 2;
} else if ( logy > KNOT_Y_LOW[ 3] && logy <= KNOT_Y_LOW[ 4]) {
cf.x = coefsLow[3];  cf.y = coefsLow[4];  cf.z = coefsLow[5];  j = 3;
} else if ( logy > KNOT_Y_LOW[ 4] && logy <= KNOT_Y_LOW[ 5]) {
cf.x = coefsLow[4];  cf.y = coefsLow[5];  cf.z = coefsLow[6];  j = 4;
} else if ( logy > KNOT_Y_LOW[ 5] && logy <= KNOT_Y_LOW[ 6]) {
cf.x = coefsLow[5];  cf.y = coefsLow[6];  cf.z = coefsLow[7];  j = 5;
} else if ( logy > KNOT_Y_LOW[ 6] && logy <= KNOT_Y_LOW[ 7]) {
cf.x = coefsLow[6];  cf.y = coefsLow[7];  cf.z = coefsLow[8];  j = 6;
}
vec3 tmp = MM * cf;
float a = tmp.x;
float b = tmp.y;
float c = tmp.z;
c = c - logy;
float d = sqrt( b * b - 4. * a * c);
float t = ( 2.0 * c) / ( -d - b);
logx = log10(C.minPoint.x) + ( t + float(j)) * KNOT_INC_LOW;
} else if ( (logy > log10(C.midPoint.y)) && (logy < log10(C.maxPoint.y)) ) {
int j;
vec3 cf;
if ( logy > KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) {
cf.x = coefsHigh[0];  cf.y = coefsHigh[1];  cf.z = coefsHigh[2];  j = 0;
} else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) {
cf.x = coefsHigh[1];  cf.y = coefsHigh[2];  cf.z = coefsHigh[3];  j = 1;
} else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) {
cf.x = coefsHigh[2];  cf.y = coefsHigh[3];  cf.z = coefsHigh[4];  j = 2;
} else if ( logy > KNOT_Y_HIGH[ 3] && logy <= KNOT_Y_HIGH[ 4]) {
cf.x = coefsHigh[3];  cf.y = coefsHigh[4];  cf.z = coefsHigh[5];  j = 3;
} else if ( logy > KNOT_Y_HIGH[ 4] && logy <= KNOT_Y_HIGH[ 5]) {
cf.x = coefsHigh[4];  cf.y = coefsHigh[5];  cf.z = coefsHigh[6];  j = 4;
} else if ( logy > KNOT_Y_HIGH[ 5] && logy <= KNOT_Y_HIGH[ 6]) {
cf.x = coefsHigh[5];  cf.y = coefsHigh[6];  cf.z = coefsHigh[7];  j = 5;
} else if ( logy > KNOT_Y_HIGH[ 6] && logy <= KNOT_Y_HIGH[ 7]) {
cf.x = coefsHigh[6];  cf.y = coefsHigh[7];  cf.z = coefsHigh[8];  j = 6;
}
vec3 tmp = MM * cf;
float a = tmp.x;
float b = tmp.y;
float c = tmp.z;
c = c - logy;
float d = sqrt( b * b - 4. * a * c);
float t = ( 2.0 * c) / ( -d - b);
logx = log10(C.midPoint.x) + ( t + float(j)) * KNOT_INC_HIGH;
} else {
logx = log10(C.maxPoint.x);
}
return pow(10.0, logx);
}

vec3 segmented_spline_c9_rev_f3( vec3 rgbPre) {
SegmentedSplineParams_c9 C = ODT_48nits();
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, C);
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, C);
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, C);
return rgbPost;
}

vec3 segmented_spline_c5_rev_f3( vec3 rgbPre) {
vec3 rgbPost;
rgbPost.x = segmented_spline_c5_rev( rgbPre.x);
rgbPost.y = segmented_spline_c5_rev( rgbPre.y);
rgbPost.z = segmented_spline_c5_rev( rgbPre.z);
return rgbPost;
}

float lin_to_ACEScc( float ya) {
if (ya <= 0.0)
return -0.3584474886;
else if (ya < pow(2.0, -15.0))
return (log2( pow(2.0, -16.0) + ya * 0.5) + 9.72) / 17.52;
else
return (log2(ya) + 9.72) / 17.52;
}

vec3 ACES_to_ACEScc( vec3 ACES) {
ACES = max( ACES, 0.0);
vec3 lin_AP1 = AP0_2_AP1_MAT * ACES;
vec3 Out;
Out.x = lin_to_ACEScc( lin_AP1.x); Out.y = lin_to_ACEScc( lin_AP1.y); Out.z = lin_to_ACEScc( lin_AP1.z);
return Out;
}

float ACEScc_to_lin( float ya) {
if (ya < -0.3013698630)
return (pow( 2.0, ya * 17.52 - 9.72) - pow( 2.0, -16.0)) * 2.0;
else
return pow( 2.0, ya * 17.52 - 9.72);
}

vec3 ACEScc_to_ACES( vec3 ACEScc) {
vec3 lin_AP1;
lin_AP1.x = ACEScc_to_lin( ACEScc.x); lin_AP1.y = ACEScc_to_lin( ACEScc.y); lin_AP1.z = ACEScc_to_lin( ACEScc.z);
vec3 ACES = AP1_2_AP0_MAT * lin_AP1;
return ACES;
}

vec3 ACES_to_ACEScg( vec3 ACES) {
ACES = max( ACES, 0.0);
vec3 ACEScg = AP0_2_AP1_MAT * ACES;
return ACEScg;
}

vec3 ACEScg_to_ACES( vec3 ACEScg) {
vec3 ACES = AP1_2_AP0_MAT * ACEScg;
return ACES;
}

float ACESproxy10_to_lin( float ya)
{
float StepsPerStop = 50.0;
float MidCVoffset = 425.0;
//float CVmin = 64.0;
//float CVmax = 940.0;
return pow( 2.0, ( ya - MidCVoffset)/StepsPerStop - 2.5);
}

vec3 ACESproxy10_to_ACES( vec3 In)
{
vec3 ACESproxy;
ACESproxy.x = In.x * 1023.0;
ACESproxy.y = In.y * 1023.0;
ACESproxy.z = In.z * 1023.0;
vec3 lin_AP1;
lin_AP1.x = ACESproxy10_to_lin( ACESproxy.x);
lin_AP1.y = ACESproxy10_to_lin( ACESproxy.y);
lin_AP1.z = ACESproxy10_to_lin( ACESproxy.z);
vec3 ACES = AP1_2_AP0_MAT * lin_AP1;
return ACES;
}

float ACESproxy12_to_lin( float ya)
{
float StepsPerStop = 200.0;
float MidCVoffset = 1700.0;
//float CVmin = 256.0;
//float CVmax = 3760.0;
return pow( 2.0, ( ya - MidCVoffset)/StepsPerStop - 2.5);
}

vec3 ACESproxy12_to_ACES( vec3 In)
{
vec3 ACESproxy;
ACESproxy.x = In.x * 4095.0;
ACESproxy.y = In.y * 4095.0;
ACESproxy.z = In.z * 4095.0;
vec3 lin_AP1;
lin_AP1.x = ACESproxy12_to_lin( ACESproxy.x);
lin_AP1.y = ACESproxy12_to_lin( ACESproxy.y);
lin_AP1.z = ACESproxy12_to_lin( ACESproxy.z);
vec3 ACES = AP1_2_AP0_MAT * lin_AP1;
return ACES;
}

float lin_to_ACESproxy10( float ya)
{
float StepsPerStop = 50.0;
float MidCVoffset = 425.0;
float CVmin = 64.0;
float CVmax = 940.0;
if (ya <= pow(2.0, -9.72))
return CVmin;
else
return max( CVmin, min( CVmax, floor( (log2(ya) + 2.5) * StepsPerStop + MidCVoffset + 0.5)) );
}

vec3 ACES_to_ACESproxy10( vec3 ACES)
{
ACES = max( ACES, 0.0); 
vec3 lin_AP1 = AP0_2_AP1_MAT * ACES;
float ACESproxy[3];
ACESproxy[0] = lin_to_ACESproxy10( lin_AP1.x );
ACESproxy[1] = lin_to_ACESproxy10( lin_AP1.y );
ACESproxy[2] = lin_to_ACESproxy10( lin_AP1.z );
vec3 Out;    
Out.x = ACESproxy[0] / 1023.0;
Out.y = ACESproxy[1] / 1023.0;
Out.z = ACESproxy[2] / 1023.0;
return Out;
}

float lin_to_ACESproxy12( float ya)
{
float StepsPerStop = 200.0;
float MidCVoffset = 1700.0;
float CVmin = 256.0;
float CVmax = 3760.0;
if (ya <= pow(2.0, -9.72))
return CVmin;
else
return max( CVmin, min( CVmax, floor( (log2(ya) + 2.5) * StepsPerStop + MidCVoffset + 0.5)));
}

vec3 ACES_to_ACESproxy12( vec3 ACES)
{
ACES = max( ACES, 0.0);
vec3 lin_AP1 = AP0_2_AP1_MAT * ACES;
float ACESproxy[3];
ACESproxy[0] = lin_to_ACESproxy12( lin_AP1.x );
ACESproxy[1] = lin_to_ACESproxy12( lin_AP1.y );
ACESproxy[2] = lin_to_ACESproxy12( lin_AP1.z );
vec3 Out;
Out.x = ACESproxy[0] / 4095.0;
Out.y = ACESproxy[1] / 4095.0;
Out.z = ACESproxy[2] / 4095.0;
return Out;
}

vec3 ADX10_to_ACES( vec3 ADX10)
{
float[11] LUT_IN = float[11](
-0.19,
0.01,
0.028,
0.054,
0.095,
0.145,
0.22,
0.3,
0.4,
0.5,
0.6);
float[11] LUT_OUT = float[11](
-6.0,
-2.721718645,
-2.521718645,
-2.321718645,
-2.121718645,
-1.921718645,
-1.721718645,
-1.521718645,
-1.321718645,
-1.121718645,
-0.926545676714876);
vec3 adx;
adx.x = ADX10.x * 1023.0;
adx.y = ADX10.y * 1023.0;
adx.z = ADX10.z * 1023.0;
vec3 cdd = ( adx - 95.0) / 500.0;
vec3 cid = CDD_TO_CID * cdd;
vec3 logE;
if ( cid.x <= 0.6) logE.x = interpolate1D11( LUT_IN, LUT_OUT, cid.x);
if ( cid.y <= 0.6) logE.y = interpolate1D11( LUT_IN, LUT_OUT, cid.y);
if ( cid.z <= 0.6) logE.z = interpolate1D11( LUT_IN, LUT_OUT, cid.z);
if ( cid.x > 0.6) logE.x = ( 100.0 / 55.0) * cid.x - REF_PT;
if ( cid.y > 0.6) logE.y = ( 100.0 / 55.0) * cid.y - REF_PT;
if ( cid.z > 0.6) logE.z = ( 100.0 / 55.0) * cid.z - REF_PT;
vec3 exp;
exp.x = pow( 10.0, logE.x);
exp.y = pow( 10.0, logE.y);
exp.z = pow( 10.0, logE.z);
vec3 aces = EXP_TO_ACES * exp;
return aces;
}

vec3 ADX16_to_ACES( vec3 ADX16)
{
float[11] LUT_IN = float[11](
-0.19,
0.01,
0.028,
0.054,
0.095,
0.145,
0.22,
0.3,
0.4,
0.5,
0.6);
float[11] LUT_OUT = float[11](
-6.0,
-2.721718645,
-2.521718645,
-2.321718645,
-2.121718645,
-1.921718645,
-1.721718645,
-1.521718645,
-1.321718645,
-1.121718645,
-0.926545676714876);
vec3 adx;
adx.x = ADX16.x * 65535.0;
adx.y = ADX16.y * 65535.0;
adx.z = ADX16.z * 65535.0;
vec3 cdd = ( adx - 1520.0) / 8000.0;
vec3 cid = CDD_TO_CID * cdd;
vec3 logE;
if ( cid.x <= 0.6) logE.x = interpolate1D11( LUT_IN, LUT_OUT, cid.x);
if ( cid.y <= 0.6) logE.y = interpolate1D11( LUT_IN, LUT_OUT, cid.y);
if ( cid.z <= 0.6) logE.z = interpolate1D11( LUT_IN, LUT_OUT, cid.z);
if ( cid.x > 0.6) logE.x = ( 100.0 / 55.0) * cid.x - REF_PT;
if ( cid.y > 0.6) logE.y = ( 100.0 / 55.0) * cid.y - REF_PT;
if ( cid.z > 0.6) logE.z = ( 100.0 / 55.0) * cid.z - REF_PT;
vec3 exp;
exp.x = pow( 10.0, logE.x);
exp.y = pow( 10.0, logE.y);
exp.z = pow( 10.0, logE.z);
vec3 aces = EXP_TO_ACES * exp;
return aces;
}

float normalizedLogCToRelativeExposure(float x)
{
if (x > 0.149659)
return (pow(10.0, (x - 0.385537) / 0.247189) - 0.052272) / 5.555556;
else
return (x - 0.092809) / 5.367650;
}

vec3 IDT_Alexa_v3_logC_EI800( vec3 Alexa)
{
float r_lin = normalizedLogCToRelativeExposure(Alexa.x);
float g_lin = normalizedLogCToRelativeExposure(Alexa.y);
float b_lin = normalizedLogCToRelativeExposure(Alexa.z);

vec3 aces;
aces.x = r_lin * 0.680206 + g_lin * 0.236137 + b_lin * 0.083658;
aces.y = r_lin * 0.085415 + g_lin * 1.017471 + b_lin * -0.102886;
aces.z = r_lin * 0.002057 + g_lin * -0.062563 + b_lin * 1.060506;

return aces;
}

vec3 IDT_Alexa_v3_raw_EI800_CCT6500( vec3 In)
{
//float EI = 800.0;
float black = 256.0 / 65535.0;
//float exp_factor = 0.18 / (0.01 * (400.0 / EI));
float r_lin = (In.x - black);// * exp_factor;
float g_lin = (In.y - black);// * exp_factor;
float b_lin = (In.z - black);// * exp_factor;

vec3 aces;
aces.x = r_lin * 0.809931 + g_lin * 0.162741 + b_lin * 0.027328;
aces.y = r_lin * 0.083731 + g_lin * 1.108667 + b_lin * -0.192397;
aces.z = r_lin * 0.044166 + g_lin * -0.272038 + b_lin * 1.227872;

return aces;
}

vec3 IDT_Panasonic_V35( vec3 VLog)
{
mat3 mat = mat3(vec3(0.724382758, 0.166748484, 0.108497411), vec3(0.021354009, 0.985138372, -0.006319092), vec3(-0.009234278, -0.00104295, 1.010272625) );

float rLin = vLogToLinScene(VLog.x);
float gLin = vLogToLinScene(VLog.y);
float bLin = vLogToLinScene(VLog.z);

vec3 Out;
Out.x = mat[0][0] * rLin + mat[0][1] * gLin + mat[0][2] * bLin;
Out.y = mat[1][0] * rLin + mat[1][1] * gLin + mat[1][2] * bLin;
Out.z = mat[2][0] * rLin + mat[2][1] * gLin + mat[2][2] * bLin;

return Out;
}

vec3 IDT_REDWideGamutRGB_Log3G10( vec3 log3G10)
{
float r_lin = Log3G10_to_linear(log3G10.x);
float g_lin = Log3G10_to_linear(log3G10.y);
float b_lin = Log3G10_to_linear(log3G10.z);

vec3 aces;
aces.x = r_lin * 0.785043 + g_lin * 0.083844 + b_lin * 0.131118;
aces.y = r_lin * 0.023172 + g_lin * 1.087892 + b_lin * -0.111055;
aces.z = r_lin * -0.073769 + g_lin * -0.314639 + b_lin * 1.388537;

return aces;
}

vec3 IDT_Canon_C100_A_D55( vec3 In)
{
float iR, iG, iB;
iR = (In.x * 1023.0 - 64.0) / 876.0;
iG = (In.y * 1023.0 - 64.0) / 876.0;
iB = (In.z * 1023.0 - 64.0) / 876.0;

vec3 pmtx;
pmtx.x = 1.08190037262167 * iR -0.180298701368782 * iG +0.0983983287471069 * iB
+1.9458545364518 * iR*iG -0.509539936937375 * iG*iB -0.47489567735516 * iB*iR
-0.778086752197068 * iR*iR -0.7412266070049 * iG*iG +0.557894437042701 * iB*iB
-3.27787395719078 * iR*iR*iG +0.254878417638717 * iR*iR*iB +3.45581530576474 * iR*iG*iG
+0.335471713974739 * iR*iG*iB -0.43352125478476 * iR*iB*iB -1.65050137344141 * iG*iG*iB +1.46581418175682 * iG*iB*iB
+0.944646566605676 * iR*iR*iR -0.723653099155881 * iG*iG*iG -0.371076501167857 * iB*iB*iB;

pmtx.y = -0.00858997792576314 * iR +1.00673740119621 * iG +0.00185257672955608 * iB
+0.0848736138296452 * iR*iG +0.347626906448902 * iG*iB +0.0020230274463939 * iB*iR
-0.0790508414091524 * iR*iR -0.179497582958716 * iG*iG -0.175975123357072 * iB*iB
+2.30205579706951 * iR*iR*iG -0.627257613385219 * iR*iR*iB -2.90795250918851 * iR*iG*iG
+1.37002437502321 * iR*iG*iB -0.108668158565563 * iR*iB*iB -2.21150552827555 * iG*iG*iB + 1.53315057595445 * iG*iB*iB
-0.543188706699505 * iR*iR*iR +1.63793038490376 * iG*iG*iG -0.444588616836587 * iB*iB*iB;

pmtx.z = 0.12696639806511 * iR -0.011891441127869 * iG +0.884925043062759 * iB
+1.34780279822258 * iR*iG +1.03647352257365 * iG*iB +0.459113289955922 * iB*iR
-0.878157422295268 * iR*iR -1.3066278750436 * iG*iG -0.658604313413283 * iB*iB
-1.4444077996703 * iR*iR*iG +0.556676588785173 * iR*iR*iB +2.18798497054968 * iR*iG*iG
-1.43030768398665 * iR*iG*iB -0.0388323570817641 * iR*iB*iB +2.63698573112453 * iG*iG*iB -1.66598882056039 * iG*iB*iB
+0.33450249360103 * iR*iR*iR -1.65856930730901 * iG*iG*iG +0.521956184547685 * iB*iB*iB;

vec3 lin;
lin.x = CanonLog_to_linear( pmtx.x);
lin.y = CanonLog_to_linear( pmtx.y);
lin.z = CanonLog_to_linear( pmtx.z);

vec3 aces;
aces.x = 0.561538969 * lin.x + 0.402060105 * lin.y + 0.036400926 * lin.z;
aces.y = 0.092739623 * lin.x + 0.924121198 * lin.y - 0.016860821 * lin.z;
aces.z = 0.084812961 * lin.x + 0.006373835 * lin.y + 0.908813204 * lin.z;

return aces;
}

vec3 IDT_Canon_C100_A_Tng( vec3 In)
{
float iR, iG, iB;
iR = (In.x * 1023.0 - 64.0) / 876.0;
iG = (In.y * 1023.0 - 64.0) / 876.0;
iB = (In.z * 1023.0 - 64.0) / 876.0;

vec3 pmtx;
pmtx.x = 0.963803004454899 * iR - 0.160722202570655 * iG + 0.196919198115756 * iB
+2.03444685639819 * iR*iG - 0.442676931451021 * iG*iB - 0.407983781537509 * iB*iR
-0.640703323129254 * iR*iR - 0.860242798247848 * iG*iG + 0.317159977967446 * iB*iB
-4.80567080102966 * iR*iR*iG + 0.27118370397567 * iR*iR*iB + 5.1069005049557 * iR*iG*iG
+0.340895816920585 * iR*iG*iB - 0.486941738507862 * iR*iB*iB - 2.23737935753692 * iG*iG*iB + 1.96647555251297 * iG*iB*iB
+1.30204051766243 * iR*iR*iR - 1.06503117628554 * iG*iG*iG - 0.392473022667378 * iB*iB*iB;

pmtx.y = -0.0421935892309314 * iR +1.04845959175183 * iG - 0.00626600252090315 * iB
-0.106438896887216 * iR*iG + 0.362908621470781 * iG*iB + 0.118070700472261 * iB*iR
+0.0193542539838734 * iR*iR - 0.156083029543267 * iG*iG - 0.237811649496433 * iB*iB
+1.67916420582198 * iR*iR*iG - 0.632835327167897 * iR*iR*iB - 1.95984471387461 * iR*iG*iG
+0.953221464562814 * iR*iG*iB + 0.0599085176294623 * iR*iB*iB - 1.66452046236246 * iG*iG*iB + 1.14041188349761 * iG*iB*iB
-0.387552623550308 * iR*iR*iR + 1.14820099685512 * iG*iG*iG - 0.336153941411709 * iB*iB*iB;

pmtx.z = 0.170295033135028 * iR - 0.0682984448537245 * iG + 0.898003411718697 * iB
+1.22106821992399 * iR*iG + 1.60194865922925 * iG*iB + 0.377599191137124 * iB*iR
-0.825781428487531 * iR*iR - 1.44590868076749 * iG*iG - 0.928925961035344 * iB*iB
-0.838548997455852 * iR*iR*iG + 0.75809397217116 * iR*iR*iB + 1.32966795243196 * iR*iG*iG
-1.20021905668355 * iR*iG*iB - 0.254838995845129 * iR*iB*iB + 2.33232411639308 * iG*iG*iB - 1.86381505762773 * iG*iB*iB
+0.111576038956423 * iR*iR*iR - 1.12593315849766 * iG*iG*iG + 0.751693186157287 * iB*iB*iB;

vec3 lin;
lin.x = CanonLog_to_linear( pmtx.x);
lin.y = CanonLog_to_linear( pmtx.y);
lin.z = CanonLog_to_linear( pmtx.z);

vec3 aces;
aces.x = 0.566996399 * lin.x + 0.365079418 * lin.y + 0.067924183 * lin.z;
aces.y = 0.070901044 * lin.x + 0.880331008 * lin.y + 0.048767948 * lin.z;
aces.z = 0.073013542 * lin.x - 0.066540862 * lin.y + 0.99352732 * lin.z;

return aces;
}

vec3 IDT_Canon_C100mk2_A_D55( vec3 In)
{
float iR, iG, iB;
iR = (In.x * 1023.0 - 64.0) / 876.0;
iG = (In.y * 1023.0 - 64.0) / 876.0;
iB = (In.z * 1023.0 - 64.0) / 876.0;

vec3 pmtx;
pmtx.x = 1.08190037262167 * iR -0.180298701368782 * iG +0.0983983287471069 * iB
+1.9458545364518 * iR*iG -0.509539936937375 * iG*iB -0.47489567735516 * iB*iR
-0.778086752197068 * iR*iR -0.7412266070049 * iG*iG +0.557894437042701 * iB*iB
-3.27787395719078 * iR*iR*iG +0.254878417638717 * iR*iR*iB +3.45581530576474 * iR*iG*iG
+0.335471713974739 * iR*iG*iB -0.43352125478476 * iR*iB*iB -1.65050137344141 * iG*iG*iB +1.46581418175682 * iG*iB*iB
+0.944646566605676 * iR*iR*iR -0.723653099155881 * iG*iG*iG -0.371076501167857 * iB*iB*iB;

pmtx.y = -0.00858997792576314 * iR +1.00673740119621 * iG +0.00185257672955608 * iB
+0.0848736138296452 * iR*iG +0.347626906448902 * iG*iB +0.0020230274463939 * iB*iR
-0.0790508414091524 * iR*iR -0.179497582958716 * iG*iG -0.175975123357072 * iB*iB
+2.30205579706951 * iR*iR*iG -0.627257613385219 * iR*iR*iB -2.90795250918851 * iR*iG*iG
+1.37002437502321 * iR*iG*iB -0.108668158565563 * iR*iB*iB -2.21150552827555 * iG*iG*iB + 1.53315057595445 * iG*iB*iB
-0.543188706699505 * iR*iR*iR +1.63793038490376 * iG*iG*iG -0.444588616836587 * iB*iB*iB;

pmtx.z = 0.12696639806511 * iR -0.011891441127869 * iG +0.884925043062759 * iB
+1.34780279822258 * iR*iG +1.03647352257365 * iG*iB +0.459113289955922 * iB*iR
-0.878157422295268 * iR*iR -1.3066278750436 * iG*iG -0.658604313413283 * iB*iB
-1.4444077996703 * iR*iR*iG +0.556676588785173 * iR*iR*iB +2.18798497054968 * iR*iG*iG
-1.43030768398665 * iR*iG*iB -0.0388323570817641 * iR*iB*iB +2.63698573112453 * iG*iG*iB -1.66598882056039 * iG*iB*iB
+0.33450249360103 * iR*iR*iR -1.65856930730901 * iG*iG*iG +0.521956184547685 * iB*iB*iB;

vec3 lin;
lin.x = CanonLog_to_linear( (pmtx.x * 876.0 + 64.0) / 1023.0 );
lin.y = CanonLog_to_linear( (pmtx.y * 876.0 + 64.0) / 1023.0 );
lin.z = CanonLog_to_linear( (pmtx.z * 876.0 + 64.0) / 1023.0 );
vec3 aces;
aces.x = 0.561538969 * lin.x + 0.402060105 * lin.y + 0.036400926 * lin.z;
aces.y = 0.092739623 * lin.x + 0.924121198 * lin.y - 0.016860821 * lin.z;
aces.z = 0.084812961 * lin.x + 0.006373835 * lin.y + 0.908813204 * lin.z;

return aces;
}

vec3 IDT_Canon_C100mk2_A_Tng( vec3 In)
{
float iR, iG, iB;
iR = (In.x * 1023.0 - 64.0) / 876.0;
iG = (In.y * 1023.0 - 64.0) / 876.0;
iB = (In.z * 1023.0 - 64.0) / 876.0;

vec3 pmtx;
pmtx.x = 0.963803004454899 * iR -0.160722202570655 * iG +0.196919198115756 * iB
+2.03444685639819 * iR*iG -0.442676931451021 * iG*iB -0.407983781537509 * iB*iR
-0.640703323129254 * iR*iR -0.860242798247848 * iG*iG +0.317159977967446 * iB*iB
-4.80567080102966 * iR*iR*iG +0.27118370397567 * iR*iR*iB +5.1069005049557 * iR*iG*iG
+0.340895816920585 * iR*iG*iB -0.486941738507862 * iR*iB*iB -2.23737935753692 * iG*iG*iB +1.96647555251297 * iG*iB*iB
+1.30204051766243 * iR*iR*iR -1.06503117628554 * iG*iG*iG -0.392473022667378 * iB*iB*iB;

pmtx.y = -0.0421935892309314 * iR +1.04845959175183 * iG -0.00626600252090315 * iB
-0.106438896887216 * iR*iG +0.362908621470781 * iG*iB +0.118070700472261 * iB*iR
+0.0193542539838734 * iR*iR -0.156083029543267 * iG*iG -0.237811649496433 * iB*iB
+1.67916420582198 * iR*iR*iG -0.632835327167897 * iR*iR*iB -1.95984471387461 * iR*iG*iG
+0.953221464562814 * iR*iG*iB +0.0599085176294623 * iR*iB*iB -1.66452046236246 * iG*iG*iB +1.14041188349761 * iG*iB*iB
-0.387552623550308 * iR*iR*iR +1.14820099685512 * iG*iG*iG -0.336153941411709 * iB*iB*iB;

pmtx.z = 0.170295033135028 * iR -0.0682984448537245 * iG +0.898003411718697 * iB
+1.22106821992399 * iR*iG +1.60194865922925 * iG*iB +0.377599191137124 * iB*iR
-0.825781428487531 * iR*iR -1.44590868076749 * iG*iG -0.928925961035344 * iB*iB
-0.838548997455852 * iR*iR*iG +0.75809397217116 * iR*iR*iB +1.32966795243196 * iR*iG*iG
-1.20021905668355 * iR*iG*iB -0.254838995845129 * iR*iB*iB +2.33232411639308 * iG*iG*iB -1.86381505762773 * iG*iB*iB
+0.111576038956423 * iR*iR*iR -1.12593315849766 * iG*iG*iG +0.751693186157287 * iB*iB*iB;

vec3 lin;
lin.x = CanonLog_to_linear( (pmtx.x * 876.0 + 64.0) / 1023.0 );
lin.y = CanonLog_to_linear( (pmtx.y * 876.0 + 64.0) / 1023.0 );
lin.z = CanonLog_to_linear( (pmtx.z * 876.0 + 64.0) / 1023.0 );
vec3 aces;
aces.x = 0.566996399 * lin.x + 0.365079418 * lin.y + 0.067924183 * lin.z;
aces.y = 0.070901044 * lin.x + 0.880331008 * lin.y + 0.048767948 * lin.z;
aces.z = 0.073013542 * lin.x - 0.066540862 * lin.y + 0.99352732 * lin.z;

return aces;
}

vec3 IDT_Canon_C300_A_D55( vec3 In)
{
float iR, iG, iB;
iR = (In.x * 1023.0 - 64.0) / 876.0;
iG = (In.y * 1023.0 - 64.0) / 876.0;
iB = (In.z * 1023.0 - 64.0) / 876.0;

vec3 pmtx;
pmtx.x = 1.08190037262167 * iR -0.180298701368782 * iG +0.0983983287471069 * iB
+1.9458545364518 * iR*iG -0.509539936937375 * iG*iB -0.47489567735516 * iB*iR
-0.778086752197068 * iR*iR -0.7412266070049 * iG*iG +0.557894437042701 * iB*iB
-3.27787395719078 * iR*iR*iG +0.254878417638717 * iR*iR*iB +3.45581530576474 * iR*iG*iG
+0.335471713974739 * iR*iG*iB -0.43352125478476 * iR*iB*iB -1.65050137344141 * iG*iG*iB +1.46581418175682 * iG*iB*iB
+0.944646566605676 * iR*iR*iR -0.723653099155881 * iG*iG*iG -0.371076501167857 * iB*iB*iB;

pmtx.y = -0.00858997792576314 * iR +1.00673740119621 * iG +0.00185257672955608 * iB
+0.0848736138296452 * iR*iG +0.347626906448902 * iG*iB +0.0020230274463939 * iB*iR
-0.0790508414091524 * iR*iR -0.179497582958716 * iG*iG -0.175975123357072 * iB*iB
+2.30205579706951 * iR*iR*iG -0.627257613385219 * iR*iR*iB -2.90795250918851 * iR*iG*iG
+1.37002437502321 * iR*iG*iB -0.108668158565563 * iR*iB*iB -2.21150552827555 * iG*iG*iB + 1.53315057595445 * iG*iB*iB
-0.543188706699505 * iR*iR*iR +1.63793038490376 * iG*iG*iG -0.444588616836587 * iB*iB*iB;

pmtx.z = 0.12696639806511 * iR -0.011891441127869 * iG +0.884925043062759 * iB
+1.34780279822258 * iR*iG +1.03647352257365 * iG*iB +0.459113289955922 * iB*iR
-0.878157422295268 * iR*iR -1.3066278750436 * iG*iG -0.658604313413283 * iB*iB
-1.4444077996703 * iR*iR*iG +0.556676588785173 * iR*iR*iB +2.18798497054968 * iR*iG*iG
-1.43030768398665 * iR*iG*iB -0.0388323570817641 * iR*iB*iB +2.63698573112453 * iG*iG*iB -1.66598882056039 * iG*iB*iB
+0.33450249360103 * iR*iR*iR -1.65856930730901 * iG*iG*iG +0.521956184547685 * iB*iB*iB;

vec3 lin;
lin.x = CanonLog_to_linear( pmtx.x);
lin.y = CanonLog_to_linear( pmtx.y);
lin.z = CanonLog_to_linear( pmtx.z);

vec3 aces;
aces.x = 0.561538969 * lin.x + 0.402060105 * lin.y + 0.036400926 * lin.z;
aces.y = 0.092739623 * lin.x + 0.924121198 * lin.y - 0.016860821 * lin.z;
aces.z = 0.084812961 * lin.x + 0.006373835 * lin.y + 0.908813204 * lin.z;

return aces;
}

vec3 IDT_Canon_C300_A_Tng( vec3 In)
{
float iR, iG, iB;
iR = (In.x * 1023.0 - 64.0) / 876.0;
iG = (In.y * 1023.0 - 64.0) / 876.0;
iB = (In.z * 1023.0 - 64.0) / 876.0;

vec3 pmtx;
pmtx.x = 0.963803004454899 * iR -0.160722202570655 * iG +0.196919198115756 * iB
+2.03444685639819 * iR*iG -0.442676931451021 * iG*iB -0.407983781537509 * iB*iR
-0.640703323129254 * iR*iR -0.860242798247848 * iG*iG +0.317159977967446 * iB*iB
-4.80567080102966 * iR*iR*iG +0.27118370397567 * iR*iR*iB +5.1069005049557 * iR*iG*iG
+0.340895816920585 * iR*iG*iB -0.486941738507862 * iR*iB*iB -2.23737935753692 * iG*iG*iB +1.96647555251297 * iG*iB*iB
+1.30204051766243 * iR*iR*iR -1.06503117628554 * iG*iG*iG -0.392473022667378 * iB*iB*iB;

pmtx.y = -0.0421935892309314 * iR +1.04845959175183 * iG -0.00626600252090315 * iB
-0.106438896887216 * iR*iG +0.362908621470781 * iG*iB +0.118070700472261 * iB*iR
+0.0193542539838734 * iR*iR -0.156083029543267 * iG*iG -0.237811649496433 * iB*iB
+1.67916420582198 * iR*iR*iG -0.632835327167897 * iR*iR*iB -1.95984471387461 * iR*iG*iG
+0.953221464562814 * iR*iG*iB +0.0599085176294623 * iR*iB*iB -1.66452046236246 * iG*iG*iB +1.14041188349761 * iG*iB*iB
-0.387552623550308 * iR*iR*iR +1.14820099685512 * iG*iG*iG -0.336153941411709 * iB*iB*iB;

pmtx.z = 0.170295033135028 * iR -0.0682984448537245 * iG +0.898003411718697 * iB
+1.22106821992399 * iR*iG +1.60194865922925 * iG*iB +0.377599191137124 * iB*iR
-0.825781428487531 * iR*iR -1.44590868076749 * iG*iG -0.928925961035344 * iB*iB
-0.838548997455852 * iR*iR*iG +0.75809397217116 * iR*iR*iB +1.32966795243196 * iR*iG*iG
-1.20021905668355 * iR*iG*iB -0.254838995845129 * iR*iB*iB +2.33232411639308 * iG*iG*iB -1.86381505762773 * iG*iB*iB
+0.111576038956423 * iR*iR*iR -1.12593315849766 * iG*iG*iG +0.751693186157287 * iB*iB*iB;

vec3 lin;
lin.x = CanonLog_to_linear( pmtx.x);
lin.y = CanonLog_to_linear( pmtx.y);
lin.z = CanonLog_to_linear( pmtx.z);

vec3 aces;
aces.x = 0.566996399 * lin.x +0.365079418 * lin.y + 0.067924183 * lin.z;
aces.y = 0.070901044 * lin.x +0.880331008 * lin.y + 0.048767948 * lin.z;
aces.z = 0.073013542 * lin.x -0.066540862 * lin.y + 0.99352732 * lin.z;

return aces;
}

vec3 IDT_Canon_C500_A_D55( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023 - 64) / 876;
CLogIRE.y = (In.y * 1023 - 64) / 876;
CLogIRE.z = (In.z * 1023 - 64) / 876;

vec3 lin;
lin.x = CanonLog_to_linear( CLogIRE.x);
lin.y = CanonLog_to_linear( CLogIRE.y);
lin.z = CanonLog_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.561538969 * lin.x +0.402060105 * lin.y + 0.036400926 * lin.z;
aces.y = 0.092739623 * lin.x +0.924121198 * lin.y - 0.016860821 * lin.z;
aces.z = 0.084812961 * lin.x +0.006373835 * lin.y + 0.908813204 * lin.z;

return aces;
}

vec3 IDT_Canon_C500_A_Tng( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023 - 64) / 876;
CLogIRE.y = (In.y * 1023 - 64) / 876;
CLogIRE.z = (In.z * 1023 - 64) / 876;

vec3 lin;
lin.x = CanonLog_to_linear( CLogIRE.x);
lin.y = CanonLog_to_linear( CLogIRE.y);
lin.z = CanonLog_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.566996399 * lin.x +0.365079418 * lin.y + 0.067924183 * lin.z;
aces.y = 0.070901044 * lin.x +0.880331008 * lin.y + 0.048767948 * lin.z;
aces.z = 0.073013542 * lin.x -0.066540862 * lin.y + 0.99352732 * lin.z;

return aces;
}

vec3 IDT_Canon_C500_B_D55( vec3 In)
{
float iR, iG, iB;
iR = (In.x * 1023.0 - 64.0) / 876.0;
iG = (In.y * 1023.0 - 64.0) / 876.0;
iB = (In.z * 1023.0 - 64.0) / 876.0;

vec3 pmtx;
pmtx.x = 1.08190037262167 * iR -0.180298701368782 * iG +0.0983983287471069 * iB
+1.9458545364518 * iR*iG -0.509539936937375 * iG*iB -0.47489567735516 * iB*iR
-0.778086752197068 * iR*iR -0.7412266070049 * iG*iG +0.557894437042701 * iB*iB
-3.27787395719078 * iR*iR*iG +0.254878417638717 * iR*iR*iB +3.45581530576474 * iR*iG*iG
+0.335471713974739 * iR*iG*iB -0.43352125478476 * iR*iB*iB -1.65050137344141 * iG*iG*iB + 1.46581418175682 * iG*iB*iB
+0.944646566605676 * iR*iR*iR -0.723653099155881 * iG*iG*iG -0.371076501167857 * iB*iB*iB;

pmtx.y = -0.00858997792576314 * iR +1.00673740119621 * iG +0.00185257672955608 * iB
+0.0848736138296452 * iR*iG +0.347626906448902 * iG*iB +0.0020230274463939 * iB*iR
-0.0790508414091524 * iR*iR -0.179497582958716 * iG*iG -0.175975123357072 * iB*iB
+2.30205579706951 * iR*iR*iG -0.627257613385219 * iR*iR*iB -2.90795250918851 * iR*iG*iG
+1.37002437502321 * iR*iG*iB -0.108668158565563 * iR*iB*iB -2.21150552827555 * iG*iG*iB + 1.53315057595445 * iG*iB*iB
-0.543188706699505 * iR*iR*iR +1.63793038490376 * iG*iG*iG -0.444588616836587 * iB*iB*iB;

pmtx.z = 0.12696639806511 * iR -0.011891441127869 * iG +0.884925043062759 * iB
+1.34780279822258 * iR*iG +1.03647352257365 * iG*iB +0.459113289955922 * iB*iR
-0.878157422295268 * iR*iR -1.3066278750436 * iG*iG -0.658604313413283 * iB*iB
-1.4444077996703 * iR*iR*iG +0.556676588785173 * iR*iR*iB +2.18798497054968 * iR*iG*iG
-1.43030768398665 * iR*iG*iB -0.0388323570817641 * iR*iB*iB +2.63698573112453 * iG*iG*iB - 1.66598882056039 * iG*iB*iB
+0.33450249360103 * iR*iR*iR -1.65856930730901 * iG*iG*iG +0.521956184547685 * iB*iB*iB;

vec3 lin;
lin.x = CanonLog_to_linear( pmtx.x);
lin.y = CanonLog_to_linear( pmtx.y);
lin.z = CanonLog_to_linear( pmtx.z);

vec3 aces;
aces.x = 0.561538969 * lin.x + 0.402060105 * lin.y + 0.036400926 * lin.z;
aces.y = 0.092739623 * lin.x + 0.924121198 * lin.y - 0.016860821 * lin.z;
aces.z = 0.084812961 * lin.x + 0.006373835 * lin.y + 0.908813204 * lin.z;

return aces;
}

vec3 IDT_Canon_C500_B_Tng( vec3 In)
{
float iR, iG, iB;
iR = (In.x * 1023.0 - 64.0) / 876.0;
iG = (In.y * 1023.0 - 64.0) / 876.0;
iB = (In.z * 1023.0 - 64.0) / 876.0;

vec3 pmtx;
pmtx.x = 0.963803004454899 * iR -0.160722202570655 * iG +0.196919198115756 * iB
+2.03444685639819 * iR*iG -0.442676931451021 * iG*iB -0.407983781537509 * iB*iR
-0.640703323129254 * iR*iR -0.860242798247848 * iG*iG +0.317159977967446 * iB*iB
-4.80567080102966 * iR*iR*iG +0.27118370397567 * iR*iR*iB +5.1069005049557 * iR*iG*iG
+0.340895816920585 * iR*iG*iB -0.486941738507862 * iR*iB*iB -2.23737935753692 * iG*iG*iB + 1.96647555251297 * iG*iB*iB
+1.30204051766243 * iR*iR*iR -1.06503117628554 * iG*iG*iG -0.392473022667378 * iB*iB*iB;

pmtx.y = -0.0421935892309314 * iR +1.04845959175183 * iG -0.00626600252090315 * iB
-0.106438896887216 * iR*iG +0.362908621470781 * iG*iB +0.118070700472261 * iB*iR
+0.0193542539838734 * iR*iR -0.156083029543267 * iG*iG -0.237811649496433 * iB*iB
+1.67916420582198 * iR*iR*iG -0.632835327167897 * iR*iR*iB -1.95984471387461 * iR*iG*iG
+0.953221464562814 * iR*iG*iB +0.0599085176294623 * iR*iB*iB -1.66452046236246 * iG*iG*iB + 1.14041188349761 * iG*iB*iB
-0.387552623550308 * iR*iR*iR +1.14820099685512 * iG*iG*iG -0.336153941411709 * iB*iB*iB;

pmtx.z = 0.170295033135028 * iR -0.0682984448537245 * iG +0.898003411718697 * iB
+1.22106821992399 * iR*iG +1.60194865922925 * iG*iB +0.377599191137124 * iB*iR
-0.825781428487531 * iR*iR -1.44590868076749 * iG*iG -0.928925961035344 * iB*iB
-0.838548997455852 * iR*iR*iG +0.75809397217116 * iR*iR*iB +1.32966795243196 * iR*iG*iG
-1.20021905668355 * iR*iG*iB -0.254838995845129 * iR*iB*iB +2.33232411639308 * iG*iG*iB - 1.86381505762773 * iG*iB*iB
+0.111576038956423 * iR*iR*iR -1.12593315849766 * iG*iG*iG +0.751693186157287 * iB*iB*iB;

vec3 lin;
lin.x = CanonLog_to_linear( pmtx.x);
lin.y = CanonLog_to_linear( pmtx.y);
lin.z = CanonLog_to_linear( pmtx.z);

vec3 aces;
aces.x = 0.566996399 * lin.x +0.365079418 * lin.y + 0.067924183 * lin.z;
aces.y = 0.070901044 * lin.x +0.880331008 * lin.y + 0.048767948 * lin.z;
aces.z = 0.073013542 * lin.x -0.066540862 * lin.y + 0.99352732 * lin.z;

return aces;
}

vec3 IDT_Canon_C500_CinemaGamut_A_D55( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.763064455 * lin.x + 0.149021161 * lin.y + 0.087914384 * lin.z;
aces.y = 0.003657457 * lin.x + 1.10696038 * lin.y - 0.110617837 * lin.z;
aces.z = -0.009407794 * lin.x - 0.218383305 * lin.y + 1.227791099 * lin.z;

return aces;
}

vec3 IDT_Canon_C500_CinemaGamut_A_Tng( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.817416293 * lin.x + 0.090755698 * lin.y + 0.091828009 * lin.z;
aces.y = -0.035361374 * lin.x + 1.065690585 * lin.y - 0.030329211 * lin.z;
aces.z = 0.010390366 * lin.x - 0.299271107 * lin.y + 1.288880741 * lin.z;

return aces;
}

vec3 IDT_Canon_C500_DCI_P3_A_D55( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.607160575 * lin.x + 0.299507286 * lin.y + 0.093332140 * lin.z;
aces.y = 0.004968120 * lin.x + 1.050982224 * lin.y - 0.055950343 * lin.z;
aces.z = -0.007839939 * lin.x + 0.000809127 * lin.y + 1.007030813 * lin.z;

return aces;
}

vec3 IDT_Canon_C500_DCI_P3_A_Tng( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.650279125 * lin.x + 0.253880169 * lin.y + 0.095840706 * lin.z;
aces.y = -0.026137986 * lin.x + 1.017900530 * lin.y + 0.008237456 * lin.z;
aces.z = 0.007757558 * lin.x - 0.063081669 * lin.y + 1.055324110 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog_BT2020_D_D55( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.678891151 * lin.x + 0.158868422 * lin.y + 0.162240427 * lin.z;
aces.y = 0.045570831 * lin.x + 0.860712772 * lin.y + 0.093716397 * lin.z;
aces.z = -0.000485710 * lin.x + 0.025060196 * lin.y + 0.975425515 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog_BT2020_D_Tng( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.724488568 * lin.x + 0.115140904 * lin.y + 0.160370529 * lin.z;
aces.y = 0.010659276 * lin.x + 0.839605344 * lin.y + 0.149735380 * lin.z;
aces.z = 0.014560161 * lin.x - 0.028562057 * lin.y + 1.014001897 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_D55( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.763064455 * lin.x + 0.149021161 * lin.y + 0.087914384 * lin.z;
aces.y = 0.003657457 * lin.x + 1.10696038 * lin.y - 0.110617837 * lin.z;
aces.z = -0.009407794 * lin.x - 0.218383305 * lin.y + 1.227791099 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog_CinemaGamut_C_Tng( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.817416293 * lin.x + 0.090755698 * lin.y + 0.091828009 * lin.z;
aces.y = -0.035361374 * lin.x + 1.065690585 * lin.y - 0.030329211 * lin.z;
aces.z = 0.010390366 * lin.x - 0.299271107 * lin.y + 1.288880741 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog2_BT2020_B_D55( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog2_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog2_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog2_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.678891151 * lin.x + 0.158868422 * lin.y + 0.162240427 * lin.z;
aces.y = 0.045570831 * lin.x + 0.860712772 * lin.y + 0.093716397 * lin.z;
aces.z = -0.000485710 * lin.x + 0.025060196 * lin.y + 0.975425515 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog2_BT2020_B_Tng( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog2_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog2_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog2_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.724488568 * lin.x + 0.115140904 * lin.y + 0.160370529 * lin.z;
aces.y = 0.010659276 * lin.x + 0.839605344 * lin.y + 0.149735380 * lin.z;
aces.z = 0.014560161 * lin.x - 0.028562057 * lin.y + 1.014001897 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_D55( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog2_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog2_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog2_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.763064455 * lin.x + 0.149021161 * lin.y + 0.087914384 * lin.z;
aces.y = 0.003657457 * lin.x + 1.10696038 * lin.y - 0.110617837 * lin.z;
aces.z = -0.009407794 * lin.x - 0.218383305 * lin.y + 1.227791099 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog2_CinemaGamut_A_Tng( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog2_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog2_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog2_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.817416293 * lin.x + 0.090755698 * lin.y + 0.091828009 * lin.z;
aces.y = -0.035361374 * lin.x + 1.065690585 * lin.y - 0.030329211 * lin.z;
aces.z = 0.010390366 * lin.x - 0.299271107 * lin.y + 1.288880741 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog3_BT2020_F_D55( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog3_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog3_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog3_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.678891151 * lin.x + 0.158868422 * lin.y + 0.162240427 * lin.z;
aces.y = 0.045570831 * lin.x + 0.860712772 * lin.y + 0.093716397 * lin.z;
aces.z = -0.000485710 * lin.x + 0.025060196 * lin.y + 0.975425515 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog3_BT2020_F_Tng( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog3_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog3_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog3_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.724488568 * lin.x + 0.115140904 * lin.y + 0.160370529 * lin.z;
aces.y = 0.010659276 * lin.x + 0.839605344 * lin.y + 0.149735380 * lin.z;
aces.z = 0.014560161 * lin.x - 0.028562057 * lin.y + 1.014001897 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_D55( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog3_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog3_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog3_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.763064455 * lin.x + 0.149021161 * lin.y + 0.087914384 * lin.z;
aces.y = 0.003657457 * lin.x + 1.10696038 * lin.y - 0.110617837 * lin.z;
aces.z = -0.009407794 * lin.x - 0.218383305 * lin.y + 1.227791099 * lin.z;

return aces;
}

vec3 IDT_Canon_C300mk2_CanonLog3_CinemaGamut_E_Tng( vec3 In)
{
vec3 CLogIRE;
CLogIRE.x = (In.x * 1023.0 - 64.0) / 876.0;
CLogIRE.y = (In.y * 1023.0 - 64.0) / 876.0;
CLogIRE.z = (In.z * 1023.0 - 64.0) / 876.0;

vec3 lin;
lin.x = 0.9 * CanonLog3_to_linear( CLogIRE.x);
lin.y = 0.9 * CanonLog3_to_linear( CLogIRE.y);
lin.z = 0.9 * CanonLog3_to_linear( CLogIRE.z);

vec3 aces;
aces.x = 0.817416293 * lin.x + 0.090755698 * lin.y + 0.091828009 * lin.z;
aces.y = -0.035361374 * lin.x + 1.065690585 * lin.y - 0.030329211 * lin.z;
aces.z = 0.010390366 * lin.x - 0.299271107 * lin.y + 1.288880741 * lin.z;

return aces;
}

vec3 IDT_Sony_SLog1_SGamut_10( vec3 In)
{
mat3 SGAMUT_TO_ACES_MTX = mat3( vec3( 0.754338638, 0.021198141, -0.009756991 ), vec3( 0.133697046, 1.005410934, 0.004508563 ), vec3( 0.111968437, -0.026610548, 1.005253201 ) );

float B = 64.0;
float AB = 90.0;
float W = 940.0;

vec3 SLog;
SLog.x = In.x * 1023.0;
SLog.y = In.y * 1023.0;
SLog.z = In.z * 1023.0;

vec3 lin;
lin.x = SLog1_to_lin( SLog.x, B, AB, W);
lin.y = SLog1_to_lin( SLog.y, B, AB, W);
lin.z = SLog1_to_lin( SLog.z, B, AB, W);

vec3 aces = SGAMUT_TO_ACES_MTX * lin;

return aces;
}

vec3 IDT_Sony_SLog1_SGamut_12( vec3 In)
{
mat3 SGAMUT_TO_ACES_MTX = mat3( vec3( 0.754338638, 0.021198141, -0.009756991 ), vec3( 0.133697046, 1.005410934, 0.004508563 ), vec3( 0.111968437, -0.026610548, 1.005253201 ) );

float B = 256.0;
float AB = 360.0;
float W = 3760.0;

vec3 SLog;
SLog.x = In.x * 4095.0;
SLog.y = In.y * 4095.0;
SLog.z = In.z * 4095.0;

vec3 lin;
lin.x = SLog1_to_lin( SLog.x, B, AB, W);
lin.y = SLog1_to_lin( SLog.y, B, AB, W);
lin.z = SLog1_to_lin( SLog.z, B, AB, W);

vec3 aces = SGAMUT_TO_ACES_MTX * lin;

return aces;
}

vec3 IDT_Sony_SLog2_SGamut_Daylight_10( vec3 In)
{
mat3 SGAMUT_DAYLIGHT_TO_ACES_MTX = mat3( vec3( 0.8764457030, 0.0774075345, 0.0573564351), vec3( 0.0145411681, 0.9529571767, -0.1151066335), vec3( 0.1090131290, -0.0303647111, 1.0577501984) );

float B = 64.0;
float AB = 90.0;
float W = 940.0;

vec3 SLog;
SLog.x = In.x * 1023.0;
SLog.y = In.y * 1023.0;
SLog.z = In.z * 1023.0;

vec3 lin;
lin.x = SLog2_to_lin( SLog.x, B, AB, W);
lin.y = SLog2_to_lin( SLog.y, B, AB, W);
lin.z = SLog2_to_lin( SLog.z, B, AB, W);

vec3 aces = SGAMUT_DAYLIGHT_TO_ACES_MTX * lin;

return aces;
}

vec3 IDT_Sony_SLog2_SGamut_Daylight_12( vec3 In)
{
mat3 SGAMUT_DAYLIGHT_TO_ACES_MTX = mat3(vec3(0.8764457030, 0.0774075345, 0.0573564351), vec3(0.0145411681, 0.9529571767, -0.1151066335), vec3(0.1090131290, -0.0303647111, 1.0577501984));

float B = 256.0;
float AB = 360.0;
float W = 3760.0;

vec3 SLog;
SLog.x = In.x * 4095.0;
SLog.y = In.y * 4095.0;
SLog.z = In.z * 4095.0;

vec3 lin;
lin.x = SLog2_to_lin( SLog.x, B, AB, W);
lin.y = SLog2_to_lin( SLog.y, B, AB, W);
lin.z = SLog2_to_lin( SLog.z, B, AB, W);

vec3 aces = SGAMUT_DAYLIGHT_TO_ACES_MTX * lin;

return aces;
}

vec3 IDT_Sony_SLog2_SGamut_Tungsten_10( vec3 In)
{
mat3 SGAMUT_TUNG_TO_ACES_MTX = mat3( vec3( 1.0110238740, 0.1011994504, 0.0600766530), vec3( -0.1362526051, 0.9562196265, -0.1010185315), vec3( 0.1252287310, -0.0574190769, 1.0409418785) );

float B = 64.0;
float AB = 90.0;
float W = 940.0;

vec3 SLog;
SLog.x = In.x * 1023.0;
SLog.y = In.y * 1023.0;
SLog.z = In.z * 1023.0;

vec3 lin;
lin.x = SLog2_to_lin( SLog.x, B, AB, W);
lin.y = SLog2_to_lin( SLog.y, B, AB, W);
lin.z = SLog2_to_lin( SLog.z, B, AB, W);

vec3 aces = SGAMUT_TUNG_TO_ACES_MTX * lin;

return aces;
}

vec3 IDT_Sony_SLog2_SGamut_Tungsten_12( vec3 In)
{
mat3 SGAMUT_TUNG_TO_ACES_MTX = mat3( vec3( 1.0110238740, 0.1011994504, 0.0600766530), vec3( -0.1362526051, 0.9562196265, -0.1010185315), vec3( 0.1252287310, -0.0574190769, 1.0409418785) );

float B = 256.0;
float AB = 360.0;
float W = 3760.0;

vec3 SLog;
SLog.x = In.x * 4095.0;
SLog.y = In.y * 4095.0;
SLog.z = In.z * 4095.0;

vec3 lin;
lin.x = SLog2_to_lin( SLog.x, B, AB, W);
lin.y = SLog2_to_lin( SLog.y, B, AB, W);
lin.z = SLog2_to_lin( SLog.z, B, AB, W);

vec3 aces = SGAMUT_TUNG_TO_ACES_MTX * lin;

return aces;
}

vec3 IDT_Sony_SLog3_SGamut3( vec3 SLog3)
{
mat3 matrixCoef = mat3( vec3(0.7529825954, 0.0217076974, -0.0094160528), vec3(0.1433702162, 1.0153188355, 0.0033704179), vec3(0.1036471884, -0.0370265329, 1.0060456349) );

vec3 linear;
linear.x = SLog3_to_linear( SLog3.x );
linear.y = SLog3_to_linear( SLog3.y );
linear.z = SLog3_to_linear( SLog3.z );

vec3 aces = matrixCoef * linear;

return aces;
}

vec3 IDT_Sony_SLog3_SGamut3Cine( vec3 SLog3)
{
mat3 matrixCoef = mat3( vec3(0.6387886672, -0.0039159060, -0.0299072021), vec3(0.2723514337, 1.0880732309, -0.0264325799), vec3(0.0888598991, -0.0841573249, 1.0563397820) );

vec3 linear;
linear.x = SLog3_to_linear( SLog3.x );
linear.y = SLog3_to_linear( SLog3.y );
linear.z = SLog3_to_linear( SLog3.z );

vec3 aces = matrixCoef * linear;

return aces;
}

float Y_2_linCV( float Y, float Ymax, float Ymin)
{
return (Y - Ymin) / (Ymax - Ymin);
}

float linCV_2_Y( float linCV, float Ymax, float Ymin)
{
return linCV * (Ymax - Ymin) + Ymin;
}

vec3 Y_2_linCV_f3( vec3 Y, float Ymax, float Ymin)
{
vec3 linCV;
linCV.x = Y_2_linCV( Y.x, Ymax, Ymin); linCV.y = Y_2_linCV( Y.y, Ymax, Ymin); linCV.z = Y_2_linCV( Y.z, Ymax, Ymin);
return linCV;
}

vec3 linCV_2_Y_f3( vec3 linCV, float Ymax, float Ymin)
{
vec3 Y;
Y.x = linCV_2_Y( linCV.x, Ymax, Ymin); Y.y = linCV_2_Y( linCV.y, Ymax, Ymin); Y.z = linCV_2_Y( linCV.z, Ymax, Ymin);
return Y;
}

vec3 darkSurround_to_dimSurround( vec3 linearCV)
{
vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
vec3 xyY = XYZ_2_xyY(XYZ);
xyY.z = max( xyY.z, 0.0);
xyY.z = pow( xyY.z, DIM_SURROUND_GAMMA);
XYZ = xyY_2_XYZ(xyY);
return XYZtoRGB(AP1) * XYZ;
}

vec3 dimSurround_to_darkSurround( vec3 linearCV)
{
vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
vec3 xyY = XYZ_2_xyY(XYZ);
xyY.z = max( xyY.z, 0.0);
xyY.z = pow( xyY.z, 1.0 / DIM_SURROUND_GAMMA);
XYZ = xyY_2_XYZ(xyY);
return XYZtoRGB(AP1) * XYZ;
}

float roll_white_fwd( float In, float new_wht, float width)
{
float x0 = -1.0;
float x1 = x0 + width;
float y0 = -new_wht;
float y1 = x1;
float m1 = (x1 - x0);
float a = y0 - y1 + m1;
float b = 2.0 * ( y1 - y0) - m1;
float c = y0;
float t = (-In - x0) / (x1 - x0);
float Out = 0.0;
if ( t < 0.0)
Out = -(t * b + c);
else if ( t > 1.0)
Out = In;
else
Out = -(( t * a + b) * t + c);
return Out;
}

float roll_white_rev( float In, float new_wht, float width)
{
float x0 = -1.0;
float x1 = x0 + width;
float y0 = -new_wht;
float y1 = x1;
float m1 = (x1 - x0);
float a = y0 - y1 + m1;
float b = 2.0 * ( y1 - y0) - m1;
float c = y0;
float Out = 0.0;
if ( -In < y0)
Out = -x0;
else if ( -In > y1)
Out = In;
else {
c = c + In;
float discrim = sqrt( b * b - 4.0 * a * c);
float t = ( 2.0 * c) / ( -discrim - b);
Out = -(( t * ( x1 - x0)) + x0);
}
return Out;
}

float lookup_ACESmin( float minLum )
{
mat2 minTable = mat2( vec2(log10(MIN_LUM_RRT), MIN_STOP_RRT ), vec2( log10(MIN_LUM_SDR), MIN_STOP_SDR ) );
return 0.18 * pow( 2.0, interpolate1D( minTable, log10( minLum)));
}

float lookup_ACESmax( float maxLum )
{
mat2 maxTable = mat2( vec2(log10(MAX_LUM_SDR), MAX_STOP_SDR ), vec2( log10(MAX_LUM_RRT), MAX_STOP_RRT ) );
return 0.18 * pow( 2.0, interpolate1D( maxTable, log10( maxLum)));
}

float5 init_coefsLow( TsPoint TsPointLow, TsPoint TsPointMid)
{
float5 coefsLow;
float knotIncLow = (log10(TsPointMid.x) - log10(TsPointLow.x)) / 3.0;
coefsLow.x = (TsPointLow.slope * (log10(TsPointLow.x) - 0.5 * knotIncLow)) + ( log10(TsPointLow.y) - TsPointLow.slope * log10(TsPointLow.x));
coefsLow.y = (TsPointLow.slope * (log10(TsPointLow.x) + 0.5 * knotIncLow)) + ( log10(TsPointLow.y) - TsPointLow.slope * log10(TsPointLow.x));
coefsLow.w = (TsPointMid.slope * (log10(TsPointMid.x) - 0.5 * knotIncLow)) + ( log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x));
coefsLow.m = (TsPointMid.slope * (log10(TsPointMid.x) + 0.5 * knotIncLow)) + ( log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x));
mat2 bendsLow = mat2( vec2(MIN_STOP_RRT, 0.18), vec2(MIN_STOP_SDR, 0.35) );
float pctLow = interpolate1D( bendsLow, log2(TsPointLow.x / 0.18));
coefsLow.z = log10(TsPointLow.y) + pctLow * (log10(TsPointMid.y) - log10(TsPointLow.y));
return coefsLow;
}

float5 init_coefsHigh( TsPoint TsPointMid, TsPoint TsPointMax)
{
float5 coefsHigh;
float knotIncHigh = (log10(TsPointMax.x) - log10(TsPointMid.x)) / 3.0;
coefsHigh.x = (TsPointMid.slope * (log10(TsPointMid.x) - 0.5 * knotIncHigh)) + ( log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x));
coefsHigh.y = (TsPointMid.slope * (log10(TsPointMid.x) + 0.5 * knotIncHigh)) + ( log10(TsPointMid.y) - TsPointMid.slope * log10(TsPointMid.x));
coefsHigh.w = (TsPointMax.slope * (log10(TsPointMax.x) - 0.5 * knotIncHigh)) + ( log10(TsPointMax.y) - TsPointMax.slope * log10(TsPointMax.x));
coefsHigh.m = (TsPointMax.slope * (log10(TsPointMax.x) + 0.5 * knotIncHigh)) + ( log10(TsPointMax.y) - TsPointMax.slope * log10(TsPointMax.x));
mat2 bendsHigh = mat2( vec2(MAX_STOP_SDR, 0.89), vec2(MAX_STOP_RRT, 0.90) );
float pctHigh = interpolate1D( bendsHigh, log2(TsPointMax.x / 0.18));
coefsHigh.z = log10(TsPointMid.y) + pctHigh*(log10(TsPointMax.y) - log10(TsPointMid.y));
return coefsHigh;
}

float shift( float In, float expShift)
{
return pow(2.0, (log2(In) - expShift));
}

TsParams init_TsParams( float minLum, float maxLum, float expShift)
{
TsPoint MIN_PT = TsPoint( lookup_ACESmin(minLum), minLum, 0.0);
TsPoint MID_PT = TsPoint( 0.18, 4.8, 1.55);
TsPoint MAX_PT = TsPoint( lookup_ACESmax(maxLum), maxLum, 0.0);
float5 cLow;
cLow = init_coefsLow( MIN_PT, MID_PT);
float5 cHigh;
cHigh = init_coefsHigh( MID_PT, MAX_PT);
MIN_PT.x = shift(lookup_ACESmin(minLum),expShift);
MID_PT.x = shift(0.18, expShift);
MAX_PT.x = shift(lookup_ACESmax(maxLum),expShift);
TsParams P = TsParams( TsPoint(MIN_PT.x, MIN_PT.y, MIN_PT.slope), TsPoint(MID_PT.x, MID_PT.y, MID_PT.slope),
TsPoint(MAX_PT.x, MAX_PT.y, MAX_PT.slope), float6(cLow.x, cLow.y, cLow.z, cLow.w, cLow.m, cLow.m),
float6(cHigh.x, cHigh.y, cHigh.z, cHigh.w, cHigh.m, cHigh.m) );
return P;
}

float ssts( float x, TsParams C)
{
const int N_KNOTS_LOW = 4;
const int N_KNOTS_HIGH = 4;
float logx = log10( max(x, 1e-10));
float logy = 0.0;
float coefsLow[6];
coefsLow[0] = C.coefsLow.a;coefsLow[1] = C.coefsLow.b;coefsLow[2] = C.coefsLow.c;
coefsLow[3] = C.coefsLow.d;coefsLow[4] = C.coefsLow.e;coefsLow[5] = C.coefsLow.f;
float coefsHigh[6];
coefsHigh[0] = C.coefsHigh.a;coefsHigh[1] = C.coefsHigh.b;coefsHigh[2] = C.coefsHigh.c;
coefsHigh[3] = C.coefsHigh.d;coefsHigh[4] = C.coefsHigh.e;coefsHigh[5] = C.coefsHigh.f;
if ( logx <= log10(C.Min.x) ) {
logy = logx * C.Min.slope + ( log10(C.Min.y) - C.Min.slope * log10(C.Min.x) );
} else if (( logx > log10(C.Min.x) ) && ( logx < log10(C.Mid.x) )) {
float knot_coord = (N_KNOTS_LOW - 1) * (logx - log10(C.Min.x)) / (log10(C.Mid.x) - log10(C.Min.x));
int j = int(knot_coord);
float t = knot_coord - j;
vec3 cf = vec3( coefsLow[ j], coefsLow[ j + 1], coefsLow[ j + 2]);
vec3 monomials = vec3( t * t, t, 1.0 );
logy = dot( monomials, MM * cf);
} else if (( logx >= log10(C.Mid.x) ) && ( logx < log10(C.Max.x) )) {
float knot_coord = (N_KNOTS_HIGH - 1) * (logx - log10(C.Mid.x)) / (log10(C.Max.x) - log10(C.Mid.x));
int j = int(knot_coord);
float t = knot_coord - j;
vec3 cf = vec3( coefsHigh[ j], coefsHigh[ j + 1], coefsHigh[ j + 2]);
vec3 monomials = vec3( t * t, t, 1.0 );
logy = dot( monomials, MM * cf);
} else {
logy = logx * C.Max.slope + ( log10(C.Max.y) - C.Max.slope * log10(C.Max.x) );
}
return pow10(logy);
}

float inv_ssts( float y, TsParams C)
{
const int N_KNOTS_LOW = 4;
const int N_KNOTS_HIGH = 4;
float KNOT_INC_LOW = (log10(C.Mid.x) - log10(C.Min.x)) / (N_KNOTS_LOW - 1.0);
float KNOT_INC_HIGH = (log10(C.Max.x) - log10(C.Mid.x)) / (N_KNOTS_HIGH - 1.0);
float KNOT_Y_LOW[ N_KNOTS_LOW];
float coefsLow[6];
coefsLow[0] = C.coefsLow.a;coefsLow[1] = C.coefsLow.b;coefsLow[2] = C.coefsLow.c;
coefsLow[3] = C.coefsLow.d;coefsLow[4] = C.coefsLow.e;coefsLow[5] = C.coefsLow.f;
float coefsHigh[6];
coefsHigh[0] = C.coefsHigh.a;coefsHigh[1] = C.coefsHigh.b;coefsHigh[2] = C.coefsHigh.c;
coefsHigh[3] = C.coefsHigh.d;coefsHigh[4] = C.coefsHigh.e;coefsHigh[5] = C.coefsHigh.f;
for (int i = 0; i < N_KNOTS_LOW; i = i + 1) {
KNOT_Y_LOW[ i] = ( coefsLow[i] + coefsLow[i + 1]) / 2.0;
};
float KNOT_Y_HIGH[ N_KNOTS_HIGH];
for (int i = 0; i < N_KNOTS_HIGH; i = i + 1) {
KNOT_Y_HIGH[ i] = ( coefsHigh[i] + coefsHigh[i + 1]) / 2.0;
};
float logy = log10( max(y, 1e-10));
float logx;
if (logy <= log10(C.Min.y)) {
logx = log10(C.Min.x);
} else if ( (logy > log10(C.Min.y)) && (logy <= log10(C.Mid.y)) ) {
int j = 0;
vec3 cf = vec3(0.0, 0.0, 0.0);
if ( logy > KNOT_Y_LOW[ 0] && logy <= KNOT_Y_LOW[ 1]) {
cf.x = coefsLow[0]; cf.y = coefsLow[1]; cf.z = coefsLow[2]; j = 0;
} else if ( logy > KNOT_Y_LOW[ 1] && logy <= KNOT_Y_LOW[ 2]) {
cf.x = coefsLow[1]; cf.y = coefsLow[2]; cf.z = coefsLow[3]; j = 1;
} else if ( logy > KNOT_Y_LOW[ 2] && logy <= KNOT_Y_LOW[ 3]) {
cf.x = coefsLow[2]; cf.y = coefsLow[3]; cf.z = coefsLow[4]; j = 2;
}
vec3 tmp = MM * cf;
float a = tmp.x; float b = tmp.y; float c = tmp.z;
c = c - logy;
float d = sqrt( b * b - 4.0 * a * c);
float t = ( 2.0 * c) / ( -d - b);
logx = log10(C.Min.x) + ( t + j) * KNOT_INC_LOW;
} else if ( (logy > log10(C.Mid.y)) && (logy < log10(C.Max.y)) ) {
int j = 0;
vec3 cf = vec3(0.0, 0.0, 0.0);
if ( logy >= KNOT_Y_HIGH[ 0] && logy <= KNOT_Y_HIGH[ 1]) {
cf.x = coefsHigh[0]; cf.y = coefsHigh[1]; cf.z = coefsHigh[2]; j = 0;
} else if ( logy > KNOT_Y_HIGH[ 1] && logy <= KNOT_Y_HIGH[ 2]) {
cf.x = coefsHigh[1]; cf.y = coefsHigh[2]; cf.z = coefsHigh[3]; j = 1;
} else if ( logy > KNOT_Y_HIGH[ 2] && logy <= KNOT_Y_HIGH[ 3]) {
cf.x = coefsHigh[2]; cf.y = coefsHigh[3]; cf.z = coefsHigh[4]; j = 2;
}
vec3 tmp = MM * cf;
float a = tmp.x; float b = tmp.y; float c = tmp.z;
c = c - logy;
float d = sqrt( b * b - 4.0 * a * c);
float t = ( 2.0 * c) / ( -d - b);
logx = log10(C.Mid.x) + ( t + j) * KNOT_INC_HIGH;
} else {
logx = log10(C.Max.x);
}
return pow10(logx);
}

vec3 ssts_f3( vec3 x, TsParams C)
{
vec3 Out;
Out.x = ssts( x.x, C); Out.y = ssts( x.y, C); Out.z = ssts( x.z, C);
return Out;
}

vec3 inv_ssts_f3( vec3 x, TsParams C)
{
vec3 Out;
Out.x = inv_ssts( x.x, C); Out.y = inv_ssts( x.y, C); Out.z = inv_ssts( x.z, C);
return Out;
}

float glow_fwd( float ycIn, float glowGainIn, float glowMid)
{
float glowGainOut;
if (ycIn <= 2.0/3.0 * glowMid) {
glowGainOut = glowGainIn;
} else if ( ycIn >= 2.0 * glowMid) {
glowGainOut = 0.0;
} else {
glowGainOut = glowGainIn * (glowMid / ycIn - 1.0/2.0);
}
return glowGainOut;
}

float glow_inv( float ycOut, float glowGainIn, float glowMid)
{
float glowGainOut;
if (ycOut <= ((1.0 + glowGainIn) * 2.0/3.0 * glowMid)) {
glowGainOut = -glowGainIn / (1.0 + glowGainIn);
} else if ( ycOut >= (2.0 * glowMid)) {
glowGainOut = 0.0;
} else {
glowGainOut = glowGainIn * (glowMid / ycOut - 1.0/2.0) / (glowGainIn / 2.0 - 1.0);
}
return glowGainOut;
}

float sigmoid_shaper( float x)
{
float t = max( 1.0 - abs( x / 2.0), 0.0);
float y = 1.0 + _sign(x) * (1.0 - t * t);
return y / 2.0;
}

float cubic_basis_shaper( float x, float w) {
mat4 M = mat4(vec4( -1.0/6.0, 3.0/6.0, -3.0/6.0, 1.0/6.0 ),
vec4( 3.0/6.0, -6.0/6.0, 3.0/6.0, 0.0/6.0 ),
vec4( -3.0/6.0, 0.0/6.0, 3.0/6.0, 0.0/6.0 ),
vec4( 1.0/6.0, 4.0/6.0, 1.0/6.0, 0.0/6.0 ) );
float knots[5];
knots[0] = -w/2.0; knots[1] = -w/4.0; knots[2] = 0.0;
knots[3] = w/4.0; knots[4] = w/2.0;
float y = 0.0;
if ((x > knots[0]) && (x < knots[4])) {  
float knot_coord = (x - knots[0]) * 4.0/w;  
int j = int(knot_coord);
float t = knot_coord - float(j);
vec4 monomials = vec4( t*t*t, t*t, t, 1.0);
if ( j == 3) {
y = monomials.x * M[0][0] + monomials.y * M[1][0] + 
monomials.z * M[2][0] + monomials.w * M[3][0];
} else if ( j == 2) {
y = monomials.x * M[0][1] + monomials.y * M[1][1] + 
monomials.z * M[2][1] + monomials.w * M[3][1];
} else if ( j == 1) {
y = monomials.x * M[0][2] + monomials.y * M[1][2] + 
monomials.z * M[2][2] + monomials.w * M[3][2];
} else if ( j == 0) {
y = monomials.x * M[0][3] + monomials.y * M[1][3] + 
monomials.z * M[2][3] + monomials.w * M[3][3];
} else {
y = 0.0;}}
return y * 3.0/2.0;
}

float center_hue( float hue, float centerH)
{
float hueCentered = hue - centerH;
if (hueCentered < -180.0) hueCentered = hueCentered + 360.0;
else if (hueCentered > 180.0) hueCentered = hueCentered - 360.0;
return hueCentered;
}

float uncenter_hue( float hueCentered, float centerH)
{
float hue = hueCentered + centerH;
if (hue < 0.0) hue = hue + 360.0;
else if (hue > 360.0) hue = hue - 360.0;
return hue;
}

vec3 rrt_sweeteners( vec3 In)
{
vec3 aces = In;
float saturation = rgb_2_saturation( aces);
float ycIn = rgb_2_yc( aces);
float s = sigmoid_shaper( (saturation - 0.4) / 0.2);
float addedGlow = 1.0 + glow_fwd( ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
aces = aces * addedGlow;
float hue = rgb_2_hue( aces);
float centeredHue = center_hue( hue, RRT_RED_HUE);
float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH);
aces.x = aces.x + hueWeight * saturation * (RRT_RED_PIVOT - aces.x) * (1.0 - RRT_RED_SCALE);
aces = max( aces, 0.0);
vec3 rgbPre = AP0_2_AP1_MAT * aces;
rgbPre = max( rgbPre, 0.0);
rgbPre = calc_sat_adjust_matrix( RRT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])) * rgbPre;
return rgbPre;
}

vec3 inv_rrt_sweeteners( vec3 In)
{
vec3 rgbPost = In;
rgbPost = invert_f33(calc_sat_adjust_matrix( RRT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1]))) * rgbPost;
rgbPost = max( rgbPost, 0.0);
vec3 aces = AP1_2_AP0_MAT * rgbPost;
aces = max( aces, 0.0);
float hue = rgb_2_hue( aces);
float centeredHue = center_hue( hue, RRT_RED_HUE);
float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH);
float minChan;
if (centeredHue < 0.0) {
minChan = aces.y;
} else {
minChan = aces.z;
}
float a = hueWeight * (1.0 - RRT_RED_SCALE) - 1.0;
float b = aces.x - hueWeight * (RRT_RED_PIVOT + minChan) * (1.0 - RRT_RED_SCALE);
float c = hueWeight * RRT_RED_PIVOT * minChan * (1.0 - RRT_RED_SCALE);
aces.x = ( -b - sqrt( b * b - 4.0 * a * c)) / ( 2.0 * a);
float saturation = rgb_2_saturation( aces);
float ycOut = rgb_2_yc( aces);
float s = sigmoid_shaper( (saturation - 0.4) / 0.2);
float reducedGlow = 1.0 + glow_inv( ycOut, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
aces = aces * reducedGlow;
return aces;
}

vec3 limit_to_primaries( vec3 XYZ, Chromaticities LIMITING_PRI)
{
mat3 XYZ_2_LIMITING_PRI_MAT = XYZtoRGB( LIMITING_PRI);
mat3 LIMITING_PRI_2_XYZ_MAT = RGBtoXYZ( LIMITING_PRI);
vec3 rgb = XYZ_2_LIMITING_PRI_MAT * XYZ;
vec3 limitedRgb = clamp( rgb, 0.0, 1.0);
return LIMITING_PRI_2_XYZ_MAT * limitedRgb;
}

vec3 dark_to_dim( vec3 XYZ)
{
vec3 xyY = XYZ_2_xyY(XYZ);
xyY.z = max( xyY.z, 0.0);
xyY.z = pow( xyY.z, DIM_SURROUND_GAMMA);
return xyY_2_XYZ(xyY);
}

vec3 dim_to_dark( vec3 XYZ)
{
vec3 xyY = XYZ_2_xyY(XYZ);
xyY.z = max( xyY.z, 0.0);
xyY.z = pow( xyY.z, 1.0 / DIM_SURROUND_GAMMA);
return xyY_2_XYZ(xyY);
}

vec3 outputTransform
(
vec3 In,
float Y_MIN,
float Y_MID,
float Y_MAX,
Chromaticities DISPLAY_PRI,
Chromaticities LIMITING_PRI,
int EOTF,
int SURROUND,
bool STRETCH_BLACK,
bool D60_SIM,
bool LEGAL_RANGE
)
{
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
TsParams PARAMS_DEFAULT = init_TsParams( Y_MIN, Y_MAX, 0.0);
float expShift = log2(inv_ssts(Y_MID, PARAMS_DEFAULT)) - log2(0.18);
TsParams PARAMS = init_TsParams( Y_MIN, Y_MAX, expShift);
vec3 rgbPre = rrt_sweeteners(In);
vec3 rgbPost = ssts_f3(rgbPre, PARAMS);
vec3 linearCV = Y_2_linCV_f3( rgbPost, Y_MAX, Y_MIN);
vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
if (SURROUND == 0) {
} else if (SURROUND == 1) {
if ((EOTF == 1) || (EOTF == 2) || (EOTF == 3)) {
XYZ = dark_to_dim( XYZ);
}
} else if (SURROUND == 2) {
}
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
if (D60_SIM == false) {
if ((DISPLAY_PRI.white.x != AP0.white.x) && (DISPLAY_PRI.white.y != AP0.white.y)) {
XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;
}
}
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
if (D60_SIM == true) {
float SCALE = 1.0;
if ((DISPLAY_PRI.white.x == 0.3127) && (DISPLAY_PRI.white.y == 0.329)) {
SCALE = 0.96362;
}
else if ((DISPLAY_PRI.white.x == 0.314) && (DISPLAY_PRI.white.y == 0.351)) {
linearCV.x = roll_white_fwd( linearCV.x, 0.918, 0.5);
linearCV.y = roll_white_fwd( linearCV.y, 0.918, 0.5);
linearCV.z = roll_white_fwd( linearCV.z, 0.918, 0.5);
SCALE = 0.96;
}
linearCV = linearCV * SCALE;
}
linearCV = max( linearCV, 0.0);
vec3 outputCV;
if (EOTF == 0) {
if (STRETCH_BLACK == true) {
outputCV = Y_2_ST2084_f3( max( linCV_2_Y_f3(linearCV, Y_MAX, 0.0), 0.0) );
} else {
outputCV = Y_2_ST2084_f3( linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN) );
}
} else if (EOTF == 1) {
outputCV = bt1886_r_f3( linearCV, 2.4, 1.0, 0.0);
} else if (EOTF == 2) {
outputCV = moncurve_r_f3( linearCV, 2.4, 0.055);
} else if (EOTF == 3) {
outputCV = pow_f3( linearCV, 1.0/2.6);
} else if (EOTF == 4) {
outputCV = linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN);
} else if (EOTF == 5) {
if (STRETCH_BLACK == true) {
outputCV = Y_2_ST2084_f3( max( linCV_2_Y_f3(linearCV, Y_MAX, 0.0), 0.0) );
}
else {
outputCV = Y_2_ST2084_f3( linCV_2_Y_f3(linearCV, Y_MAX, Y_MIN) );
}
outputCV = ST2084_2_HLG_1000nits_f3( outputCV);
}
if (LEGAL_RANGE == true) {
outputCV = fullRange_to_smpteRange_f3( outputCV);
}
return outputCV;
}

vec3 invOutputTransform
(
vec3 In,
float Y_MIN,
float Y_MID,
float Y_MAX,
Chromaticities DISPLAY_PRI,
Chromaticities LIMITING_PRI,
int EOTF,
int SURROUND,
bool STRETCH_BLACK,
bool D60_SIM,
bool LEGAL_RANGE
)
{
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI);
TsParams PARAMS_DEFAULT = init_TsParams( Y_MIN, Y_MAX, 0.0);
float expShift = log2(inv_ssts(Y_MID, PARAMS_DEFAULT)) - log2(0.18);
TsParams PARAMS = init_TsParams( Y_MIN, Y_MAX, expShift);
vec3 outputCV = In;
if (LEGAL_RANGE == true) {
outputCV = smpteRange_to_fullRange_f3( outputCV);
}
vec3 linearCV;
if (EOTF == 0) {
if (STRETCH_BLACK == true) {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, 0.0);
} else {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, Y_MIN);
}
} else if (EOTF == 1) {
linearCV = bt1886_f_f3( outputCV, 2.4, 1.0, 0.0);
} else if (EOTF == 2) {
linearCV = moncurve_f_f3( outputCV, 2.4, 0.055);
} else if (EOTF == 3) {
linearCV = pow_f3( outputCV, 2.6);
} else if (EOTF == 4) {
linearCV = Y_2_linCV_f3( outputCV, Y_MAX, Y_MIN);
} else if (EOTF == 5) {
outputCV = HLG_2_ST2084_1000nits_f3( outputCV);
if (STRETCH_BLACK == true) {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, 0.0);
} else {
linearCV = Y_2_linCV_f3( ST2084_2_Y_f3( outputCV), Y_MAX, Y_MIN);
}
}
if (D60_SIM == true) {
float SCALE = 1.0;
if ((DISPLAY_PRI.white.x == 0.3127) && (DISPLAY_PRI.white.y == 0.329)) {
SCALE = 0.96362;
linearCV = linearCV * (1.0 / SCALE);
}
else if ((DISPLAY_PRI.white.x == 0.314) && (DISPLAY_PRI.white.y == 0.351)) {
SCALE = 0.96;
linearCV.x = roll_white_rev( linearCV.x / SCALE, 0.918, 0.5);
linearCV.y = roll_white_rev( linearCV.y / SCALE, 0.918, 0.5);
linearCV.z = roll_white_rev( linearCV.z / SCALE, 0.918, 0.5);
}
}
vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
if (D60_SIM == false) {
if ((DISPLAY_PRI.white.x != AP0.white.x) && (DISPLAY_PRI.white.y != AP0.white.y)) {
XYZ = invert_f33(calculate_cat_matrix(AP0.white, REC709_PRI.white)) * XYZ;
}
}
if (SURROUND == 0) {
} else if (SURROUND == 1) {

if ((EOTF == 1) || (EOTF == 2) || (EOTF == 3)) {
XYZ = dim_to_dark( XYZ);
}
} else if (SURROUND == 2) {
}
linearCV = XYZtoRGB(AP1) * XYZ;
vec3 rgbPost = linCV_2_Y_f3( linearCV, Y_MAX, Y_MIN);
vec3 rgbPre = inv_ssts_f3( rgbPost, PARAMS);
vec3 aces = inv_rrt_sweeteners( rgbPre);
return aces;
}

vec3 InvODT_Rec709( vec3 outputCV) {
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(REC709_PRI);
float DISPGAMMA = 2.4;
float L_W = 1.0;
float L_B = 0.0;
vec3 linearCV = bt1886_f_f3( outputCV, DISPGAMMA, L_W, L_B);
vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
XYZ = XYZ * invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white));
linearCV = XYZtoRGB(AP1) * XYZ;
linearCV = invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1]))) * linearCV;
vec3 rgbPre = linCV_2_Y_f3( linearCV, 48.0, pow(10.0, log10(0.02)));
vec3 rgbPost;
rgbPost = segmented_spline_c9_rev_f3( rgbPre);
vec3 oces = AP1_2_AP0_MAT * rgbPost;
return oces;
}

vec3 InvODT_sRGB( vec3 outputCV) {
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(REC709_PRI);
float DISPGAMMA = 2.4;
float OFFSET = 0.055;
vec3 linearCV;
linearCV = moncurve_f_f3( outputCV, DISPGAMMA, OFFSET);
vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
XYZ = XYZ * invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white));
linearCV = XYZtoRGB(AP1) * XYZ;
linearCV = invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1]))) * linearCV;
vec3 rgbPre = linCV_2_Y_f3( linearCV, 48.0, pow(10.0, log10(0.02)));
vec3 rgbPost;
rgbPost = segmented_spline_c9_rev_f3( rgbPre);
vec3 oces = AP1_2_AP0_MAT * rgbPost;
return oces;
}

vec3 IDT_sRGB( vec3 rgb) {
vec3 aces;
aces = InvODT_sRGB(rgb);
aces = segmented_spline_c5_rev_f3( aces);
aces = max(aces, 0.0);
return aces;
}

vec3 IDT_rec709( vec3 rgb) {
vec3 aces;
aces = InvODT_Rec709(rgb);
aces = segmented_spline_c5_rev_f3( aces);
aces = max(aces, 0.0);
return aces;
}

float lin_to_ACEScct( float In)
{
if (In <= X_BRK)
return A * In + B;
else
return (log2(In) + 9.72) / 17.52;
}

float ACEScct_to_lin( float In)
{
if (In > Y_BRK)
return pow( 2.0, In * 17.52 - 9.72);
else
return (In - B) / A;
}

vec3 ACES_to_ACEScct( vec3 In)
{
vec3 ap1_lin = AP0_2_AP1_MAT * In;
vec3 acescct;
acescct.x = lin_to_ACEScct( ap1_lin.x); acescct.y = lin_to_ACEScct( ap1_lin.y); acescct.z = lin_to_ACEScct( ap1_lin.z);
return acescct;
}

vec3 ACEScct_to_ACES( vec3 In)
{
vec3 ap1_lin;
ap1_lin.x = ACEScct_to_lin( In.x); ap1_lin.y = ACEScct_to_lin( In.y); ap1_lin.z = ACEScct_to_lin( In.z);
return AP1_2_AP0_MAT * ap1_lin;
}

vec3 ASCCDL_inACEScct
(
vec3 acesIn,
vec3 SLOPE,
vec3 OFFSET,
vec3 POWER,
float SAT
)
{
vec3 acescct = ACES_to_ACEScct( acesIn);
acescct.x = pow( clamp( (acescct.x * SLOPE.x) + OFFSET.x, 0.0, 1.0), 1.0 / POWER.x);
acescct.y = pow( clamp( (acescct.y * SLOPE.y) + OFFSET.y, 0.0, 1.0), 1.0 / POWER.y);
acescct.z = pow( clamp( (acescct.z * SLOPE.z) + OFFSET.z, 0.0, 1.0), 1.0 / POWER.z);
float luma = 0.2126 * acescct.x + 0.7152 * acescct.y + 0.0722 * acescct.z;
float satClamp = max(SAT, 0.0);
acescct.x = luma + satClamp * (acescct.x - luma);
acescct.y = luma + satClamp * (acescct.y - luma);
acescct.z = luma + satClamp * (acescct.z - luma);
return ACEScct_to_ACES( acescct);
}

vec3 gamma_adjust_linear( vec3 rgbIn, float GAMMA, float PIVOT)
{
float SCALAR = PIVOT / pow( PIVOT, GAMMA);
vec3 rgbOut = rgbIn;
if (rgbIn.x > 0.0) rgbOut.x = pow( rgbIn.x, GAMMA) * SCALAR;
if (rgbIn.y > 0.0) rgbOut.y = pow( rgbIn.y, GAMMA) * SCALAR;
if (rgbIn.z > 0.0) rgbOut.z = pow( rgbIn.z, GAMMA) * SCALAR;
return rgbOut;
}

vec3 sat_adjust( vec3 rgbIn, float SAT_FACTOR)
{
vec3 RGB2Y = vec3(RGBtoXYZ( REC709_PRI)[0][1], RGBtoXYZ( REC709_PRI)[1][1], RGBtoXYZ( REC709_PRI)[2][1]);
mat3 SAT_MAT = calc_sat_adjust_matrix( SAT_FACTOR, RGB2Y);
return SAT_MAT * rgbIn;
}

vec3 rgb_2_yab( vec3 rgb)
{
vec3 yab = mat3(vec3(1.0/3.0, 1.0/2.0, 0.0), vec3(1.0/3.0, -1.0/4.0, sqrt3over4), vec3(1.0/3.0, -1.0/4.0, -sqrt3over4)) * rgb;
return yab;
}

vec3 yab_2_rgb( vec3 yab)
{
vec3 rgb = invert_f33(mat3(vec3(1.0/3.0, 1.0/2.0, 0.0), vec3(1.0/3.0, -1.0/4.0, sqrt3over4), vec3(1.0/3.0, -1.0/4.0, -sqrt3over4))) * yab;
return rgb;
}

vec3 yab_2_ych(vec3 yab)
{
vec3 ych = yab;
float yb = yab.y * yab.y + yab.z * yab.z;
ych.y = sqrt(yb);
ych.z = atan(yab.z, yab.y) * (180.0 / 3.14159265358979323846264338327950288);
if (ych.z < 0.0) ych.z += 360.0;
return ych;
}

vec3 ych_2_yab( vec3 ych )
{
vec3 yab;
yab.x = ych.x;
float h = ych.z * (3.14159265358979323846264338327950288 / 180.0);
yab.y = ych.y * cos(h);
yab.z = ych.y * sin(h);
return yab;
}

vec3 rgb_2_ych( vec3 rgb)
{
return yab_2_ych( rgb_2_yab( rgb));
}

vec3 ych_2_rgb( vec3 ych)
{
return yab_2_rgb( ych_2_yab( ych));
}

vec3 scale_C_at_H( vec3 rgb, float centerH, float widthH, float percentC)
{
vec3 new_rgb = rgb;
vec3 ych = rgb_2_ych( rgb);
if (ych.y > 0.0) {
float centeredHue = center_hue( ych.z, centerH);
float f_H = cubic_basis_shaper( centeredHue, widthH);
if (f_H > 0.0) {
vec3 new_ych = ych;
new_ych.y = ych.y * (f_H * (percentC - 1.0) + 1.0);
new_rgb = ych_2_rgb( new_ych);
} else {
new_rgb = rgb;
}}
return new_rgb;
}

vec3 rotate_H_in_H( vec3 rgb, float centerH, float widthH, float degreesShift)
{
vec3 ych = rgb_2_ych( rgb);
vec3 new_ych = ych;
float centeredHue = center_hue( ych.z, centerH);
float f_H = cubic_basis_shaper( centeredHue, widthH);
float old_hue = centeredHue;
float new_hue = centeredHue + degreesShift;
mat2 table = mat2( vec2(0.0, old_hue), vec2(1.0, new_hue) );
float blended_hue = interpolate1D( table, f_H);
if (f_H > 0.0) new_ych.z = uncenter_hue(blended_hue, centerH);
return ych_2_rgb( new_ych);
}

vec3 scale_C( vec3 rgb, float percentC)
{
vec3 ych = rgb_2_ych( rgb);
ych.y = ych.y * percentC;
return ych_2_rgb( ych);
}

vec3 overlay_f3( vec3 a, vec3 b)
{
float LUMA_CUT = lin_to_ACEScct( 0.5);
float luma = 0.2126 * a.x + 0.7152 * a.y + 0.0722 * a.z;
vec3 Out;
if (luma < LUMA_CUT) {
Out.x = 2.0 * a.x * b.x;
Out.y = 2.0 * a.y * b.y;
Out.z = 2.0 * a.z * b.z;
} else {
Out.x = 1.0 - (2.0 * (1.0 - a.x) * (1.0 - b.x));
Out.y = 1.0 - (2.0 * (1.0 - a.y) * (1.0 - b.y));
Out.z = 1.0 - (2.0 * (1.0 - a.z) * (1.0 - b.z));
}
return Out;
}

vec3 LMT_PFE( vec3 aces)
{
aces = scale_C( aces, 0.7);
vec3 SLOPE = vec3(1.0, 1.0, 0.94);
vec3 OFFSET = vec3(0.0, 0.0, 0.02);
vec3 POWER = vec3(1.0, 1.0, 1.0);
float SAT = 1.0;
aces = ASCCDL_inACEScct( aces, SLOPE, OFFSET, POWER, SAT);
aces = gamma_adjust_linear( aces, 1.5, 0.18);
aces = rotate_H_in_H( aces, 0.0, 30.0, 5.0);
aces = rotate_H_in_H( aces, 80.0, 60.0, -15.0);
aces = rotate_H_in_H( aces, 52.0, 50.0, -14.0);
aces = scale_C_at_H( aces, 45.0, 40.0, 1.4);
aces = rotate_H_in_H( aces, 190.0, 40.0, 30.0);
aces = scale_C_at_H( aces, 240.0, 120.0, 1.4);

return aces;
}

vec3 LMT_Bleach( vec3 aces)
{
vec3 a, b, blend;
a = sat_adjust( aces, 0.9);
a = a * 2.0;
b = sat_adjust( aces, 0.0);
b = gamma_adjust_linear( b, 1.2, 0.18);
a = ACES_to_ACEScct( a);
b = ACES_to_ACEScct( b);
blend = overlay_f3( a, b);
aces = ACEScct_to_ACES( blend);

return aces;
}

vec3 LMT_BlueLightArtifactFix( vec3 aces)
{
mat3 correctionMatrix = mat3(
vec3(0.9404372683, 0.0083786969, 0.0005471261 ),
vec3(-0.0183068787, 0.8286599939, -0.0008833746 ),
vec3( 0.0778696104, 0.1629613092, 1.0003362486 ) );
vec3 acesMod = correctionMatrix * aces;
return acesMod;
}

vec3 RRT( vec3 aces)
{
float saturation = rgb_2_saturation( aces);
float ycIn = rgb_2_yc( aces);
float s = sigmoid_shaper( (saturation - 0.4) / 0.2);
float addedGlow = 1.0 + glow_fwd( ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
aces = aces * addedGlow;
float hue = rgb_2_hue( aces);
float centeredHue = center_hue( hue, RRT_RED_HUE);
float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH);
aces.x = aces.x + hueWeight * saturation * (RRT_RED_PIVOT - aces.x) * (1.0 - RRT_RED_SCALE);
aces = max( aces, 0.0);
vec3 rgbPre = AP0_2_AP1_MAT * aces;
rgbPre = max( rgbPre, 0.0);
rgbPre = calc_sat_adjust_matrix( RRT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])) * rgbPre;
//rgbPre = RRT_SAT_MAT * rgbPre;
vec3 rgbPost;
rgbPost.x = segmented_spline_c5_fwd( rgbPre.x);
rgbPost.y = segmented_spline_c5_fwd( rgbPre.y);
rgbPost.z = segmented_spline_c5_fwd( rgbPre.z);
vec3 rgbOces = AP1_2_AP0_MAT * rgbPost;
return rgbOces;
}

vec3 InvRRT( vec3 oces)
{
vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c5_rev( rgbPre.x);
rgbPost.y = segmented_spline_c5_rev( rgbPre.y);
rgbPost.z = segmented_spline_c5_rev( rgbPre.z);
//rgbPost = rgbPost * invert_f33(calc_sat_adjust_matrix( RRT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])));
rgbPost = invert_f33(RRT_SAT_MAT) * rgbPost;
rgbPost = max( rgbPost, 0.0);
vec3 aces = AP1_2_AP0_MAT * rgbPost;
aces = max( aces, 0.0);
float hue = rgb_2_hue( aces);
float centeredHue = center_hue( hue, RRT_RED_HUE);
float hueWeight = cubic_basis_shaper( centeredHue, RRT_RED_WIDTH);
float minChan;
if (centeredHue < 0.0) {
minChan = aces.y;
} else {
minChan = aces.z;
}
float a = hueWeight * (1.0 - RRT_RED_SCALE) - 1.0;
float b = aces.x - hueWeight * (RRT_RED_PIVOT + minChan) * (1.0 - RRT_RED_SCALE);
float c = hueWeight * RRT_RED_PIVOT * minChan * (1.0 - RRT_RED_SCALE);
aces.x = ( -b - sqrt( b * b - 4.0 * a * c)) / ( 2.0 * a);
float saturation = rgb_2_saturation( aces);
float ycOut = rgb_2_yc( aces);
float s = sigmoid_shaper( (saturation - 0.4) / 0.2);
float reducedGlow = 1.0 + glow_inv( ycOut, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
aces = aces * reducedGlow;

return aces;
}

vec3 ODT_Rec709_100nits_dim( vec3 oces)
{
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
float DISPGAMMA = 2.4;
float L_W = 1.0;
float L_B = 0.0;
bool legalRange = false;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])) * linearCV;
vec3 XYZ = RGBtoXYZ(AP1) * linearCV;

XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;

linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);

vec3 outputCV;
outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);

if(legalRange) outputCV = fullRange_to_smpteRange_f3( outputCV);

return outputCV;
}

vec3 ODT_Rec709_D60sim_100nits_dim( vec3 oces)
{
const Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.4;
const float L_W = 1.0;
const float L_B = 0.0;
const float SCALE = 0.955;
bool legalRange = false;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

linearCV.x = min( linearCV.x, 1.0) * SCALE;
linearCV.y = min( linearCV.y, 1.0) * SCALE;
linearCV.z = min( linearCV.z, 1.0) * SCALE;
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])) * linearCV;

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);

vec3 outputCV;
outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);

if (legalRange) {
outputCV = fullRange_to_smpteRange_f3( outputCV);
}

return outputCV;
}

vec3 ODT_Rec2020_100nits_dim( vec3 oces)
{
Chromaticities DISPLAY_PRI = REC2020_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
float DISPGAMMA = 2.4;
float L_W = 1.0;
float L_B = 0.0;
bool legalRange = false;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])) * linearCV;

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);

vec3 outputCV;
outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);

if (legalRange) {
outputCV = fullRange_to_smpteRange_f3( outputCV);
}

return outputCV;
}

vec3 ODT_Rec2020_ST2084_1000nits( vec3 oces)
{
Chromaticities DISPLAY_PRI = REC2020_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_1000nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_1000nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_1000nits());
rgbPost = rgbPost -pow10(-4.4550166483);

vec3 XYZ = RGBtoXYZ(AP1) * rgbPost;
XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;
vec3 rgb = XYZ_2_DISPLAY_PRI_MAT * XYZ;
rgb = max( rgb, 0.0);
vec3 outputCV = Y_2_ST2084_f3( rgb);

return outputCV;
}

vec3 ODT_Rec2020_Rec709limited_100nits_dim( vec3 oces)
{
const Chromaticities DISPLAY_PRI = REC2020_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const Chromaticities LIMITING_PRI = REC709_PRI;
const float DISPGAMMA = 2.4;
const float L_W = 1.0;
const float L_B = 0.0;
bool legalRange = false;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])) * linearCV;

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);

vec3 outputCV;
outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);

if (legalRange) {
outputCV = fullRange_to_smpteRange_f3( outputCV);
}

return outputCV;
}

vec3 ODT_Rec2020_P3D65limited_100nits_dim( vec3 oces)
{
const Chromaticities DISPLAY_PRI = REC2020_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const Chromaticities LIMITING_PRI = P3D65_PRI;
const float DISPGAMMA = 2.4;
const float L_W = 1.0;
const float L_B = 0.0;
bool legalRange = false;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])) * linearCV;

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);

vec3 outputCV;
outputCV.x = bt1886_r( linearCV.x, DISPGAMMA, L_W, L_B);
outputCV.y = bt1886_r( linearCV.y, DISPGAMMA, L_W, L_B);
outputCV.z = bt1886_r( linearCV.z, DISPGAMMA, L_W, L_B);

if (legalRange) {
outputCV = fullRange_to_smpteRange_f3( outputCV);
}

return outputCV;
}

vec3 ODT_sRGB_D60sim_100nits_dim( vec3 oces)
{
const Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const float DISPGAMMA = 2.4;
const float OFFSET = 0.055;
const float SCALE = 0.955;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

linearCV.x = min( linearCV.x, 1.0) * SCALE;
linearCV.y = min( linearCV.y, 1.0) * SCALE;
linearCV.z = min( linearCV.z, 1.0) * SCALE;
linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])) * linearCV;

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);

vec3 outputCV;
outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET);
outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET);
outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET);

return outputCV;
}

vec3 ODT_sRGB_100nits_dim( vec3 oces) {
const Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const float DISPGAMMA = 2.4; 
const float OFFSET = 0.055;
vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());
vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, CINEMA_WHITE, CINEMA_BLACK);
linearCV.y = Y_2_linCV( rgbPost.y, CINEMA_WHITE, CINEMA_BLACK);
linearCV.z = Y_2_linCV( rgbPost.z, CINEMA_WHITE, CINEMA_BLACK);    
linearCV = darkSurround_to_dimSurround(linearCV);
linearCV = ODT_SAT_MAT * linearCV;
vec3 XYZ = AP1_2_XYZ_MAT * linearCV;
XYZ = D60_2_D65_CAT * XYZ;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);
vec3 outputCV;
outputCV = moncurve_r_f3( linearCV, DISPGAMMA, OFFSET);
return outputCV;
}

vec3 ODT_P3DCI_48nits( vec3 oces)
{
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.6;
const float NEW_WHT = 0.918;
const float ROLL_WIDTH = 0.5;
const float SCALE = 0.96;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

linearCV.x = roll_white_fwd( linearCV.x, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_fwd( linearCV.y, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_fwd( linearCV.z, NEW_WHT, ROLL_WIDTH);

linearCV.x = min( linearCV.x, NEW_WHT) * SCALE;
linearCV.y = min( linearCV.y, NEW_WHT) * SCALE;
linearCV.z = min( linearCV.z, NEW_WHT) * SCALE;

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);
vec3 outputCV = pow_f3( linearCV, 1.0 / DISPGAMMA);

return outputCV;
}

vec3 ODT_P3DCI_D60sim_48nits( vec3 oces)
{
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.6;
const float NEW_WHT = 0.918;
const float ROLL_WIDTH = 0.5;
const float SCALE = 0.96;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

linearCV.x = roll_white_fwd( linearCV.x, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_fwd( linearCV.y, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_fwd( linearCV.z, NEW_WHT, ROLL_WIDTH);

linearCV.x = min( linearCV.x, NEW_WHT) * SCALE;
linearCV.y = min( linearCV.y, NEW_WHT) * SCALE;
linearCV.z = min( linearCV.z, NEW_WHT) * SCALE;

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);
vec3 outputCV = pow_f3( linearCV, 1.0 / DISPGAMMA);

return outputCV;
}

vec3 ODT_P3DCI_D65sim_48nits( vec3 oces)
{
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.6;
const float NEW_WHT = 0.908;
const float ROLL_WIDTH = 0.5;
const float SCALE = 0.9575;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

linearCV.x = roll_white_fwd( linearCV.x, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_fwd( linearCV.y, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_fwd( linearCV.z, NEW_WHT, ROLL_WIDTH);

linearCV.x = min( linearCV.x, NEW_WHT) * SCALE;
linearCV.y = min( linearCV.y, NEW_WHT) * SCALE;
linearCV.z = min( linearCV.z, NEW_WHT) * SCALE;

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);
vec3 outputCV = pow_f3( linearCV, 1.0 / DISPGAMMA);

return outputCV;
}

vec3 ODT_P3D60_48nits( vec3 oces)
{
const Chromaticities DISPLAY_PRI = P3D60_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.6;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);
vec3 outputCV = pow_f3( linearCV, 1.0 / DISPGAMMA);

return outputCV;
}

vec3 ODT_P3D65_48nits( vec3 oces)
{
const Chromaticities DISPLAY_PRI = P3D65_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB( DISPLAY_PRI);
const float DISPGAMMA = 2.6;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);
vec3 outputCV = pow_f3( linearCV, 1.0 / DISPGAMMA);

return outputCV;
}

vec3 ODT_P3D65_D60sim_48nits( vec3 oces)
{
const Chromaticities DISPLAY_PRI = P3D65_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const float DISPGAMMA = 2.6;
const float SCALE = 0.964;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

linearCV.x = min( linearCV.x, 1.0) * SCALE;
linearCV.y = min( linearCV.y, 1.0) * SCALE;
linearCV.z = min( linearCV.z, 1.0) * SCALE;

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);
vec3 outputCV = pow_f3( linearCV, 1.0 / DISPGAMMA);

return outputCV;
}

vec3 ODT_P3D65_Rec709limited_48nits( vec3 oces)
{
const Chromaticities DISPLAY_PRI = P3D65_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
const Chromaticities LIMITING_PRI = REC709_PRI;
const float DISPGAMMA = 2.6;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);
vec3 outputCV = pow_f3( linearCV, 1.0 / DISPGAMMA);

return outputCV;
}

vec3 ODT_DCDM( vec3 oces)
{
vec3 rgbPre = AP0_2_AP1_MAT * oces;

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;

XYZ = max( XYZ, 0.0);
vec3 outputCV = dcdm_encode( XYZ);

return outputCV;
}

vec3 ODT_DCDM_P3D60limited( vec3 oces)
{
const Chromaticities LIMITING_PRI = P3D60_PRI;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
vec3 outputCV = dcdm_encode( XYZ);

return outputCV;
}

vec3 ODT_DCDM_P3D65limited( vec3 oces)
{
const Chromaticities LIMITING_PRI = P3D65_PRI;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
XYZ = limit_to_primaries( XYZ, LIMITING_PRI);
vec3 outputCV = dcdm_encode( XYZ);

return outputCV;
}

vec3 ODT_RGBmonitor_100nits_dim( vec3 oces)
{
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
float DISPGAMMA = 2.4;
float OFFSET = 0.055;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])) * linearCV;

vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
XYZ = calculate_cat_matrix( AP0.white, REC709_PRI.white) * XYZ;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);

vec3 outputCV;
outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET);
outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET);
outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET);

return outputCV;
}

vec3 ODT_RGBmonitor_D60sim_100nits_dim( vec3 oces)
{
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 XYZ_2_DISPLAY_PRI_MAT = XYZtoRGB(DISPLAY_PRI);
float DISPGAMMA = 2.4;
float OFFSET = 0.055;
float SCALE = 0.955;

vec3 rgbPre = AP0_2_AP1_MAT * oces;
vec3 rgbPost;
rgbPost.x = segmented_spline_c9_fwd( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_fwd( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_fwd( rgbPre.z, ODT_48nits());

vec3 linearCV;
linearCV.x = Y_2_linCV( rgbPost.x, 48.0, pow(10.0, log10(0.02)));
linearCV.y = Y_2_linCV( rgbPost.y, 48.0, pow(10.0, log10(0.02)));
linearCV.z = Y_2_linCV( rgbPost.z, 48.0, pow(10.0, log10(0.02)));

linearCV.x = min( linearCV.x, 1.0) * SCALE;
linearCV.y = min( linearCV.y, 1.0) * SCALE;
linearCV.z = min( linearCV.z, 1.0) * SCALE;

linearCV = darkSurround_to_dimSurround( linearCV);
linearCV = calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1])) * linearCV;
vec3 XYZ = RGBtoXYZ(AP1) * linearCV;
linearCV = XYZ_2_DISPLAY_PRI_MAT * XYZ;
linearCV = clamp( linearCV, 0.0, 1.0);

vec3 outputCV;
outputCV.x = moncurve_r( linearCV.x, DISPGAMMA, OFFSET);
outputCV.y = moncurve_r( linearCV.y, DISPGAMMA, OFFSET);
outputCV.z = moncurve_r( linearCV.z, DISPGAMMA, OFFSET);

return outputCV;
}

vec3 InvODT_Rec709_100nits_dim( vec3 outputCV)
{
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
float DISPGAMMA = 2.4;
float L_W = 1.0;
float L_B = 0.0;
bool legalRange = false;

if (legalRange) {
outputCV = smpteRange_to_fullRange_f3( outputCV);
}

vec3 linearCV;
linearCV.x = bt1886_f( outputCV.x, DISPGAMMA, L_W, L_B);
linearCV.y = bt1886_f( outputCV.y, DISPGAMMA, L_W, L_B);
linearCV.z = bt1886_f( outputCV.z, DISPGAMMA, L_W, L_B);

vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
XYZ = invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white)) * XYZ;
linearCV = XYZtoRGB(AP1) * XYZ;
linearCV = invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1]))) * linearCV;
linearCV = dimSurround_to_darkSurround( linearCV);

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_Rec709_D60sim_100nits_dim( vec3 outputCV)
{
const Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.4;
const float L_W = 1.0;
const float L_B = 0.0;
const float SCALE = 0.955;
bool legalRange = false;

if (legalRange) {
outputCV = smpteRange_to_fullRange_f3( outputCV);
}

vec3 linearCV;
linearCV.x = bt1886_f( outputCV.x, DISPGAMMA, L_W, L_B);
linearCV.y = bt1886_f( outputCV.y, DISPGAMMA, L_W, L_B);
linearCV.z = bt1886_f( outputCV.z, DISPGAMMA, L_W, L_B);

vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
linearCV = XYZtoRGB(AP1) * XYZ;
linearCV = invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1]))) * linearCV;
linearCV = dimSurround_to_darkSurround( linearCV);
linearCV.x = linearCV.x / SCALE;
linearCV.y = linearCV.y / SCALE;
linearCV.z = linearCV.z / SCALE;

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_Rec2020_100nits_dim( vec3 outputCV)
{
Chromaticities DISPLAY_PRI = REC2020_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
float DISPGAMMA = 2.4;
float L_W = 1.0;
float L_B = 0.0;
bool legalRange = false;

if (legalRange) {
outputCV = smpteRange_to_fullRange_f3( outputCV);
}

vec3 linearCV;
linearCV.x = bt1886_f( outputCV.x, DISPGAMMA, L_W, L_B);
linearCV.y = bt1886_f( outputCV.y, DISPGAMMA, L_W, L_B);
linearCV.z = bt1886_f( outputCV.z, DISPGAMMA, L_W, L_B);

vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
XYZ = invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white)) * XYZ;
linearCV = XYZtoRGB(AP1) * XYZ;
linearCV = invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1]))) * linearCV;
linearCV = dimSurround_to_darkSurround( linearCV);

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_Rec2020_ST2084_1000nits( vec3 outputCV)
{
Chromaticities DISPLAY_PRI = REC2020_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);

vec3 rgb = ST2084_2_Y_f3( outputCV);
vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * rgb;
XYZ = invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white)) * XYZ;

vec3 rgbPre = XYZtoRGB(AP1) * XYZ;
rgbPre = rgbPre - pow10(-4.4550166483);

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_1000nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_1000nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_1000nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_sRGB_D60sim_100nits_dim( vec3 outputCV)
{
const Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.4;
const float OFFSET = 0.055;
const float SCALE = 0.955;

vec3 linearCV;
linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET);
linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET);
linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET);

vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
linearCV = XYZtoRGB(AP1) * XYZ;
linearCV = invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1]))) * linearCV;
linearCV = dimSurround_to_darkSurround( linearCV);
linearCV.x = linearCV.x / SCALE;
linearCV.y = linearCV.y / SCALE;
linearCV.z = linearCV.z / SCALE;

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_sRGB_100nits_dim( vec3 outputCV)
{
const Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.4;
const float OFFSET = 0.055;

vec3 linearCV;
linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET);
linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET);
linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET);

vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
XYZ = XYZ * invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white));
linearCV = XYZtoRGB(AP1) * XYZ;
linearCV = invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1]))) * linearCV;
linearCV = dimSurround_to_darkSurround( linearCV);

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_P3DCI_48nits( vec3 outputCV)
{
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.6;
const float NEW_WHT = 0.918;
const float ROLL_WIDTH = 0.5;
const float SCALE = 0.96;

vec3 linearCV = pow_f3( outputCV, DISPGAMMA);
vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;

linearCV = XYZtoRGB(AP1) * XYZ;
linearCV.x = roll_white_rev( linearCV.x / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_rev( linearCV.y / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_rev( linearCV.z / SCALE, NEW_WHT, ROLL_WIDTH);

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_P3DCI_D60sim_48nits( vec3 outputCV)
{
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.6;
const float NEW_WHT = 0.918;
const float ROLL_WIDTH = 0.5;
const float SCALE = 0.96;

vec3 linearCV = pow_f3( outputCV, DISPGAMMA);
vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;

linearCV = XYZtoRGB(AP1) * XYZ;
linearCV.x = roll_white_rev( linearCV.x / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_rev( linearCV.y / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_rev( linearCV.z / SCALE, NEW_WHT, ROLL_WIDTH);

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_P3DCI_D65sim_48nits( vec3 outputCV)
{
const Chromaticities DISPLAY_PRI = P3DCI_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
const float DISPGAMMA = 2.6;
const float NEW_WHT = 0.908;
const float ROLL_WIDTH = 0.5;
const float SCALE = 0.9575;

vec3 linearCV = pow_f3( outputCV, DISPGAMMA);
vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;

linearCV = XYZtoRGB(AP1) * XYZ;
linearCV.x = roll_white_rev( linearCV.x / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.y = roll_white_rev( linearCV.y / SCALE, NEW_WHT, ROLL_WIDTH);
linearCV.z = roll_white_rev( linearCV.z / SCALE, NEW_WHT, ROLL_WIDTH);

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_P3D60_48nits( vec3 outputCV)
{
const Chromaticities DISPLAY_PRI = P3D60_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI);
const float DISPGAMMA = 2.6;

vec3 linearCV = pow_f3( outputCV, DISPGAMMA);
vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
linearCV = XYZtoRGB(AP1) * XYZ;

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_P3D65_48nits( vec3 outputCV)
{
const Chromaticities DISPLAY_PRI = P3D65_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI);
const float DISPGAMMA = 2.6;

vec3 linearCV = pow_f3( outputCV, DISPGAMMA);
vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
linearCV = XYZtoRGB(AP1) * XYZ;

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_P3D65_D60sim_48nits( vec3 outputCV)
{
const Chromaticities DISPLAY_PRI = P3D65_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ( DISPLAY_PRI);
const float DISPGAMMA = 2.6;
const float SCALE = 0.964;

vec3 linearCV = pow_f3( outputCV, DISPGAMMA);

vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;

linearCV = XYZtoRGB(AP1) * XYZ;
linearCV.x = linearCV.x / SCALE;
linearCV.y = linearCV.y / SCALE;
linearCV.z = linearCV.z / SCALE;

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_DCDM( vec3 outputCV)
{
vec3 XYZ = dcdm_decode( outputCV);
vec3 linearCV = XYZtoRGB(AP1) * XYZ;

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_DCDM_P3D65limited( vec3 outputCV)
{
vec3 XYZ = dcdm_decode( outputCV);
XYZ = XYZ * invert_f33(calculate_cat_matrix( AP0.white, REC709_PRI.white));
vec3 linearCV = XYZtoRGB(AP1) * XYZ;

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_RGBmonitor_100nits_dim( vec3 outputCV)
{
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
float DISPGAMMA = 2.4;
float OFFSET = 0.055;

vec3 linearCV;
linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET);
linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET);
linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET);

vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
XYZ = XYZ * invert_f33( calculate_cat_matrix( AP0.white, REC709_PRI.white));
linearCV = XYZtoRGB(AP1) * XYZ;
linearCV = invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1]))) * linearCV;
linearCV = dimSurround_to_darkSurround( linearCV);

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 InvODT_RGBmonitor_D60sim_100nits_dim( vec3 outputCV)
{
Chromaticities DISPLAY_PRI = REC709_PRI;
mat3 DISPLAY_PRI_2_XYZ_MAT = RGBtoXYZ(DISPLAY_PRI);
float DISPGAMMA = 2.4;
float OFFSET = 0.055;
float SCALE = 0.955;

vec3 linearCV;
linearCV.x = moncurve_f( outputCV.x, DISPGAMMA, OFFSET);
linearCV.y = moncurve_f( outputCV.y, DISPGAMMA, OFFSET);
linearCV.z = moncurve_f( outputCV.z, DISPGAMMA, OFFSET);

vec3 XYZ = DISPLAY_PRI_2_XYZ_MAT * linearCV;
linearCV = XYZtoRGB(AP1) * XYZ;
linearCV = invert_f33( calc_sat_adjust_matrix( ODT_SAT_FACTOR, vec3(RGBtoXYZ( AP1)[0][1], RGBtoXYZ( AP1)[1][1], RGBtoXYZ( AP1)[2][1]))) * linearCV;
linearCV = dimSurround_to_darkSurround( linearCV);
linearCV.x = linearCV.x / SCALE;
linearCV.y = linearCV.y / SCALE;
linearCV.z = linearCV.z / SCALE;

vec3 rgbPre;
rgbPre.x = linCV_2_Y( linearCV.x, 48.0, pow(10.0, log10(0.02)));
rgbPre.y = linCV_2_Y( linearCV.y, 48.0, pow(10.0, log10(0.02)));
rgbPre.z = linCV_2_Y( linearCV.z, 48.0, pow(10.0, log10(0.02)));

vec3 rgbPost;
rgbPost.x = segmented_spline_c9_rev( rgbPre.x, ODT_48nits());
rgbPost.y = segmented_spline_c9_rev( rgbPre.y, ODT_48nits());
rgbPost.z = segmented_spline_c9_rev( rgbPre.z, ODT_48nits());

vec3 oces = AP1_2_AP0_MAT * rgbPost;

return oces;
}

vec3 RRTODT_P3D65_108nits_7_2nits_ST2084( vec3 aces)
{
float Y_MIN = 0.0001;
float Y_MID = 7.2;
float Y_MAX = 108.0;
Chromaticities DISPLAY_PRI = P3D65_PRI;
Chromaticities LIMITING_PRI = P3D65_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;

vec3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );

return cv;
}

vec3 RRTODT_Rec2020_1000nits_15nits_HLG( vec3 aces)
{
float Y_MIN = 0.0001;
float Y_MID = 15.0;
float Y_MAX = 1000.0;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 5;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;

vec3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );

return cv;
}

vec3 RRTODT_Rec2020_1000nits_15nits_ST2084( vec3 aces)
{
float Y_MIN = 0.0001;
float Y_MID = 15.0;
float Y_MAX = 1000.0;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

vec3 RRTODT_Rec2020_2000nits_15nits_ST2084( vec3 aces)
{
float Y_MIN = 0.0001;
float Y_MID = 15.0;
float Y_MAX = 2000.0;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

vec3 RRTODT_Rec2020_4000nits_15nits_ST2084( vec3 aces)
{
float Y_MIN = 0.0001;
float Y_MID = 15.0;
float Y_MAX = 4000.0;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

vec3 RRTODT_Rec709_100nits_10nits_BT1886( vec3 aces)
{
float Y_MIN = 0.0001;
float Y_MID = 10.0;
float Y_MAX = 100.0;
Chromaticities DISPLAY_PRI = REC709_PRI;
Chromaticities LIMITING_PRI = REC709_PRI;
int EOTF = 1;
int SURROUND = 1;
bool STRETCH_BLACK = false;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

vec3 RRTODT_Rec709_100nits_10nits_sRGB( vec3 aces)
{
float Y_MIN = 0.0001;
float Y_MID = 10.0;
float Y_MAX = 100.0;
Chromaticities DISPLAY_PRI = REC709_PRI;
Chromaticities LIMITING_PRI = REC709_PRI;
int EOTF = 2;
int SURROUND = 1;
bool STRETCH_BLACK = false;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

vec3 InvRRTODT_P3D65_108nits_7_2nits_ST2084( vec3 aces)
{
float Y_MIN = 0.0001;
float Y_MID = 7.2;
float Y_MAX = 108.0;
Chromaticities DISPLAY_PRI = P3D65_PRI;
Chromaticities LIMITING_PRI = P3D65_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 cv = invOutputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

vec3 InvRRTODT_Rec2020_1000nits_15nits_HLG( vec3 cv)
{
float Y_MIN = 0.0001;
float Y_MID = 15.0;
float Y_MAX = 1000.0;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 5;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return aces;
}

vec3 InvRRTODT_Rec2020_1000nits_15nits_ST2084( vec3 cv)
{
float Y_MIN = 0.0001;
float Y_MID = 15.0;
float Y_MAX = 1000.0;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return aces;
}

vec3 InvRRTODT_Rec2020_2000nits_15nits_ST2084( vec3 cv)
{
float Y_MIN = 0.0001;
float Y_MID = 15.0;
float Y_MAX = 2000.0;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return aces;
}

vec3 InvRRTODT_Rec2020_4000nits_15nits_ST2084( vec3 cv)
{
float Y_MIN = 0.0001;
float Y_MID = 15.0;
float Y_MAX = 4000.0;
Chromaticities DISPLAY_PRI = REC2020_PRI;
Chromaticities LIMITING_PRI = REC2020_PRI;
int EOTF = 0;
int SURROUND = 0;
bool STRETCH_BLACK = true;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 aces = invOutputTransform( cv, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return aces;
}

vec3 InvRRTODT_Rec709_100nits_10nits_BT1886( vec3 aces)
{
float Y_MIN = 0.0001;
float Y_MID = 10.0;
float Y_MAX = 100.0;
Chromaticities DISPLAY_PRI = REC709_PRI;
Chromaticities LIMITING_PRI = REC709_PRI;
int EOTF = 1;
int SURROUND = 1;
bool STRETCH_BLACK = false;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

vec3 InvRRTODT_Rec709_100nits_10nits_sRGB( vec3 aces)
{
float Y_MIN = 0.0001;
float Y_MID = 10.0;
float Y_MAX = 100.0;
Chromaticities DISPLAY_PRI = REC709_PRI;
Chromaticities LIMITING_PRI = REC709_PRI;
int EOTF = 2;
int SURROUND = 1;
bool STRETCH_BLACK = false;
bool D60_SIM = false;
bool LEGAL_RANGE = false;
vec3 cv = outputTransform( aces, Y_MIN, Y_MID, Y_MAX, DISPLAY_PRI,
LIMITING_PRI, EOTF, SURROUND, STRETCH_BLACK, D60_SIM, LEGAL_RANGE );
return cv;
}

void main()
{
vec2 uv = gl_FragCoord.xy / vec2( adsk_result_w, adsk_result_h );
vec3 ACES = texture2D(front, uv).rgb;

if(p_IDT != 0) ACES = p_IDT == 1 ? ACEScc_to_ACES(ACES) : p_IDT == 2 ? ACEScct_to_ACES(ACES) : p_IDT == 3 ? ACEScg_to_ACES(ACES) : p_IDT == 4 ? IDT_rec709(ACES) : p_IDT == 5 ? IDT_sRGB(ACES) : p_IDT == 6 ? 
IDT_Alexa_v3_logC_EI800(ACES) : p_IDT == 7 ? IDT_REDWideGamutRGB_Log3G10(ACES) : p_IDT == 8 ? IDT_Panasonic_V35(ACES) : p_IDT == 9 ? IDT_Sony_SLog1_SGamut_10(ACES) : p_IDT == 10 ? 
IDT_Sony_SLog2_SGamut_Daylight_10(ACES) : p_IDT == 11 ? IDT_Sony_SLog2_SGamut_Tungsten_10(ACES) : p_IDT == 12 ? IDT_Sony_SLog3_SGamut3(ACES) : p_IDT == 13 ? IDT_Sony_SLog3_SGamut3Cine(ACES) : p_IDT == 14 ? ADX10_to_ACES(ACES) : ADX16_to_ACES(ACES);

if(p_Exposure != 0.0) ACES = ACES * exp2(p_Exposure);
if(p_LMT != 0) ACES = p_LMT == 1 ? LMT_PFE(ACES) : p_LMT == 2 ? LMT_Bleach(ACES) : LMT_BlueLightArtifactFix(ACES);
if(p_RRT != 0 && p_ODT != 5) ACES = RRT(ACES);
if(p_ODT != 0) ACES = p_ODT == 1 ? ODT_Rec709_100nits_dim(ACES) : p_ODT == 2 ? ODT_sRGB_100nits_dim(ACES) : p_ODT == 3 ? ODT_Rec2020_100nits_dim(ACES) : p_ODT == 4 ? ODT_P3DCI_48nits(ACES) : RRTODT_Rec709_100nits_10nits_BT1886(ACES);

gl_FragColor = vec4(ACES, 1.0);
}