//   LMT_Academy_Analytic_3 by Scott Dyer (converted for Shadertoy from original CTL by Baldavenger)
//
//   Input is linear ACES2065-1
//   Output is linear ACES2065-1

#define PI 3.141592
#define PIE 3.14159265358979323846264338327950288
#define X_BRK 0.0078125
#define Y_BRK 0.155251141552511
#define Aa 10.5402377416545
#define B 0.0729055341958355

uniform float p_Scale1 = 0.7; // colour scale (hint), min= 0., max=2.
uniform float p_Scale2 = 1; // slopeR, min=0., max=2.
uniform float p_Scale3 = 1; // slopeG, min=0., max=2.
uniform float p_Scale4 = 0.94; // slopeB, min=0., max=2.
uniform float p_Scale5 = 0; // offsetR, min=-2., max=2.
uniform float p_Scale6 = 0; // offsetG, min=-2., max=2.
uniform float p_Scale7 = 0.02; // offsetB, min=-2., max=2.
uniform float p_Scale8 = 1; // powerR, min=0., max=3.
uniform float p_Scale9 = 1; // powerG, min=0., max=3.
uniform float p_Scale10 = 1; // powerB, min=0., max=3.
uniform float p_Scale11 = 1; // sat, min=0., max=2.
uniform float p_Scale12 = 1.5; // gamma_adjust_linear, min=0., max=3.
uniform float p_Scale13 = 0.18; // pivot, min=0., max=1.
uniform float p_Scale14 = 0; // rotateHinH1, min=0., max=360.
uniform float p_Scale15 = 30; // range1, min=0., max=180.
uniform float p_Scale16 = 5; // shift1, min=-90., max=90.
uniform float p_Scale17 = 80; // rotateHinH2, min=0., max=360.
uniform float p_Scale18 = 60; // range2, min=0., max=180.
uniform float p_Scale19 = -15; // shift2, min=-90., max=90.
uniform float p_Scale20 = 52; // rotateHinH3, min=0., max=360.
uniform float p_Scale21 = 50; // range3, min=0., max=180.
uniform float p_Scale22 = -14; // shift3, min=-90., max=90.
uniform float p_Scale23 = 45; // scale_C_at_H1, min=0., max=360.
uniform float p_Scale24 = 40; // rangeC1, min=0., max=180.
uniform float p_Scale25 = 1.4; // scaleC1, min=0., max=2.
uniform float p_Scale26 = 190; // rotateHinH4, min=0., max=360.
uniform float p_Scale27 = 40; // range4, min=0., max=180.
uniform float p_Scale28 = 30; // shift4, min=-90., max=90.
uniform float p_Scale29 = 240; // scale_C_at_H2, min=0., max=360.
uniform float p_Scale30 = 120; // rangeC2, min=0., max=180.
uniform float p_Scale31 = 1.4; // scaleC2, min=0., max=2.

vec3[3] mult_f_f33(float f, vec3 AA[3])
{
for( int i = 0; i < 3; ++i )
{
AA[i].x *= f;
AA[i].y *= f;
AA[i].z *= f;
}
return AA;
}

vec3 mult_f3_f33(vec3 In, vec3 A[3])
{
vec3 Out;
Out.x = In.x * A[0].x + In.y * A[0].y + In.z * A[0].z;
Out.y = In.x * A[1].x + In.y * A[1].y + In.z * A[1].z;
Out.z = In.x * A[2].x + In.y * A[2].y + In.z * A[2].z;
return Out;
}

vec3[3] invert_f33(vec3 A[3])
{
vec3 result[3];
float det =   A[0].x * A[1].y * A[2].z
			+ A[0].y * A[1].z * A[2].x
			+ A[0].z * A[1].x * A[2].y
			- A[2].x * A[1].y * A[0].z
			- A[2].y * A[1].z * A[0].x
			- A[2].z * A[1].x * A[0].y;
			
if( det != 0.0 )
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

A =  mult_f_f33( 1.0 / det, result);
}
return A;
}

vec3 rgb_2_yab(vec3 rgb)
{
vec3 rgb_yab[3];
rgb_yab[0].x = 1.0/3.0;
rgb_yab[0].y = 1.0/2.0;
rgb_yab[0].z = 0.0;
rgb_yab[1].x = 1.0/3.0;
rgb_yab[1].y = -1.0/4.0;
rgb_yab[1].z = 0.433012701892219;
rgb_yab[2].x = 1.0/3.0;
rgb_yab[2].y = -1.0/4.0;
rgb_yab[2].z = -0.433012701892219;
vec3 yab;
yab = mult_f3_f33(rgb, rgb_yab);
return yab;
}

vec3 yab_2_ych(vec3 yab)
{
vec3 ych;
ych = yab;
float yo = yab.y * yab.y + yab.z * yab.z;
ych.y = sqrt(yo);
ych.z = atan(yab.z, yab.y) * (180.0 / PIE);
if (ych.z < 0.0)
{
ych.z += 360.0;
}
return ych;
}

vec3 ych_2_yab(vec3 ych) 
{
vec3 yab;
yab.x = ych.x;
float h = ych.z * (PIE / 180.0);
yab.y = ych.y * cos(h);
yab.z = ych.y * sin(h);
return yab;
}

vec3 yab_2_rgb(vec3 yab)
{
vec3 rgb_yab[3];
rgb_yab[0].x = 1.0/3.0;
rgb_yab[0].y = 1.0/2.0;
rgb_yab[0].z = 0.0;
rgb_yab[1].x = 1.0/3.0;
rgb_yab[1].y = -1.0/4.0;
rgb_yab[1].z = 0.433012701892219;
rgb_yab[2].x = 1.0/3.0;
rgb_yab[2].y = -1.0/4.0;
rgb_yab[2].z = -0.433012701892219;
vec3 rgb;
vec3 abc[3]; 
abc = invert_f33(rgb_yab);
rgb = mult_f3_f33(yab, abc);
return rgb;
}

vec3 scale_C(vec3 rgb, float percentC)
{
vec3 ych, yab;
yab = rgb_2_yab(rgb);
ych = yab_2_ych(yab);
ych.y *= percentC;
yab = ych_2_yab(ych);
rgb = yab_2_rgb(yab);
return rgb;
}

float lin_to_ACEScct(float In)
{
float Out;
if (In <= X_BRK){
Out = Aa * In + B;
} else {
Out = (log2(In) + 9.72) / 17.52;
}
return Out;
}

float ACEScct_to_lin(float In)
{
float Out;    
if (In > Y_BRK){
Out = pow(2.0, In * 17.52 - 9.72);
} else {
Out = (In - B) / Aa;
}
return Out;
}

vec3 ACES_to_ACEScct(vec3 In)
{
vec3 Out;

//AP0 to AP1
Out.x =  1.4514393161 * In.x + -0.2365107469 * In.y + -0.2149285693 * In.z;
Out.y = -0.0765537734 * In.x +  1.1762296998 * In.y + -0.0996759264 * In.z;
Out.z =  0.0083161484 * In.x + -0.0060324498 * In.y +  0.9977163014 * In.z;

// Linear to ACEScct
Out.x = lin_to_ACEScct(Out.x);
Out.y = lin_to_ACEScct(Out.y);
Out.z = lin_to_ACEScct(Out.z);

return Out;
}

vec3 ACEScct_to_ACES(vec3 In)
{
vec3 lin, Out;

// ACEScct to linear
lin.x = ACEScct_to_lin(In.x);
lin.y = ACEScct_to_lin(In.y);
lin.z = ACEScct_to_lin(In.z);

// AP1 to AP0
Out.x =  0.6954522414 * lin.x +  0.1406786965 * lin.y +  0.1638690622 * lin.z;
Out.y =  0.0447945634 * lin.x +  0.8596711185 * lin.y +  0.0955343182 * lin.z;
Out.z = -0.0055258826 * lin.x +  0.0040252103 * lin.y +  1.0015006723 * lin.z;

return Out;
}

vec3 ASCCDL_inACEScct
(
vec3 acesIn, 
float SLOPE[3],
float OFFSET[3],
float POWER[3],
float SAT
)
{

acesIn = ACES_to_ACEScct(acesIn);

acesIn.x = pow(clamp((acesIn.x * SLOPE[0]) + OFFSET[0], 0.0, 1.0), POWER[0]);
acesIn.y = pow(clamp((acesIn.y * SLOPE[1]) + OFFSET[1], 0.0, 1.0), POWER[1]);
acesIn.z = pow(clamp((acesIn.z * SLOPE[2]) + OFFSET[2], 0.0, 1.0), POWER[2]);

float luma = 0.2126 *acesIn.x + 0.7152 * acesIn.y + 0.0722 * acesIn.z;

float satClamp = clamp(SAT, 0.0, 10.0);    
acesIn.x = luma + satClamp * (acesIn.x - luma);
acesIn.y = luma + satClamp * (acesIn.y - luma);
acesIn.z = luma + satClamp * (acesIn.z - luma);

acesIn = ACEScct_to_ACES(acesIn);

return acesIn;
}

vec3 gamma_adjust_linear(vec3 rgbIn, float GAMMA, float PIVOT)
{
float SCALAR = PIVOT / pow(PIVOT, GAMMA);

if (rgbIn.x > 0.0){ rgbIn.x = pow(rgbIn.x, GAMMA) * SCALAR;}
if (rgbIn.y > 0.0){ rgbIn.y = pow(rgbIn.y, GAMMA) * SCALAR;}
if (rgbIn.z > 0.0){ rgbIn.z = pow(rgbIn.z, GAMMA) * SCALAR;}
return rgbIn;
}

float interpolate1D(vec2 table[2], float p, int t)
{
if( p <= table[0].x ) return table[0].y;
if( p >= table[t - 1].x ) return table[t - 1].y;

for( int i = 0; i < t - 1; ++i )
{
if( table[i].x <= p && p < table[i+1].x )
{
float s = (p - table[i].x) / (table[i+1].x - table[i].x);
return table[i].y * ( 1.0 - s ) + table[i+1].y * s;
}
}
return 0.0;
}

float cubic_basis_shaper(float x, float w)
{

vec4 M[4];
M[0].x = -1./6;
M[0].y = 3./6;
M[0].z = -3./6;
M[0].w = 1./6;
M[1].x = 3./6;
M[1].y = -6./6;
M[1].z = 3./6;
M[1].w = 0./6;
M[2].x = -3./6;
M[2].y = 0./6;
M[2].z = 3./6;
M[2].w = 0./6;
M[3].x = 1./6;
M[3].y = 4./6;
M[3].z = 1./6;
M[3].w = 0./6;
  
//float knots[5] = { -w/2.0, -w/4.0, 0.0, w/4.0, w/2.0 };
float knots[5];
knots[0] = -w/2.0;
knots[1] = -w/4.0;
knots[2] = 0.0;
knots[3] = w/4.0;
knots[4] = w/2.0;

float y = 0.0;
if ((x > knots[0]) && (x < knots[4])) {  
float knot_coord = (x - knots[0]) * 4.0/w;  
float j = knot_coord;
float t = knot_coord - j;

//float monomials[4] = { t*t*t, t*t, t, 1. };
float monomials[4];
monomials[0] = t*t*t;
monomials[1] = t*t;
monomials[2] = t;
monomials[3] = 1.;

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
y = 0.0;
}
}

return y * 3/2.0;
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

vec3 rotate_H_in_H(vec3 rgb, float centerH, float widthH, float degreesShift)
{
vec3 ych, yab;
yab = rgb_2_yab(rgb);
ych = yab_2_ych(yab);

float centeredHue = center_hue(ych.z, centerH);
float f_H = cubic_basis_shaper(centeredHue, widthH);

float old_hue = centeredHue;
float new_hue = centeredHue + degreesShift;
//vec2 table[2] = {{0.0, old_hue}, {1.0, new_hue}};
vec2 table[2];
table[0].x = 0.0;
table[0].y = old_hue;
table[1].x = 1.0;
table[1].y = new_hue;
float blended_hue = interpolate1D(table, f_H, 2);
 
if (f_H > 0.0)
{
ych.z = uncenter_hue(blended_hue, centerH);
}

yab = ych_2_yab(ych);
rgb = yab_2_rgb(yab);
return rgb;
}

vec3 scale_C_at_H
( 
vec3 rgb, 
float centerH,
float widthH,
float percentC
)
{
vec3 ych, yab, new_rgb;
new_rgb = rgb;
yab = rgb_2_yab(rgb);
ych = yab_2_ych(yab);
if (ych.y > 0.0) {
float centeredHue = center_hue(ych.z, centerH);
float f_H = cubic_basis_shaper(centeredHue, widthH);
if (f_H > 0.0) {
vec3 new_ych = ych;
new_ych.y = ych.y * (f_H * (percentC - 1.0) + 1.0);
yab = ych_2_yab(new_ych);
new_rgb = yab_2_rgb(yab);
} else { 
new_rgb = rgb; 
}
}
return new_rgb;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) 
{ 
vec2 uv = fragCoord.xy / iResolution.xy;
vec4 source = texture2D(iChannel0,uv);
vec3 Aces;
Aces.r = source.r;
Aces.g = source.g;
Aces.b = source.b;

Aces = scale_C(Aces, p_Scale1);

float SLOPE[3];
SLOPE[0] = p_Scale2;
SLOPE[1] = p_Scale3;
SLOPE[2] = p_Scale4;
float OFFSET[3];
OFFSET[0] = p_Scale5;
OFFSET[1] = p_Scale6;
OFFSET[2] = p_Scale7;
float POWER[3];
POWER[0] = 1/p_Scale8;
POWER[1] = 1/p_Scale9;
POWER[2] = 1/p_Scale10;
float SAT = p_Scale11;

Aces = ASCCDL_inACEScct(Aces, SLOPE, OFFSET, POWER, SAT);

Aces = gamma_adjust_linear(Aces, p_Scale12, p_Scale13);

Aces = rotate_H_in_H(Aces, p_Scale14, p_Scale15, p_Scale16);

Aces = rotate_H_in_H(Aces, p_Scale17, p_Scale18, p_Scale19);

Aces = rotate_H_in_H(Aces, p_Scale20, p_Scale21, p_Scale22);

Aces = scale_C_at_H(Aces, p_Scale23, p_Scale24, p_Scale25);

Aces = rotate_H_in_H(Aces, p_Scale26, p_Scale27, p_Scale28);

Aces = scale_C_at_H(Aces, p_Scale29, p_Scale30, p_Scale31);

fragColor = vec4(Aces, 1.0);
}