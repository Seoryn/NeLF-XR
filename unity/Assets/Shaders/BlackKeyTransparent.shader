Shader "Unlit/BlackKeyTransparent"
{
    // 인스펙터에서 조절 가능한 변수
    Properties{
        _MainTex ("RGB Texture", 2D) = "white" {}
        _Thresh  ("Black Threshold", Range(0,0.2)) = 0.06   // 임계값
        _Smooth  ("Edge Softness",  Range(0,0.2)) = 0.03    // 경계 softness 폭
        _Despill ("Despill Strength", Range(0,1)) = 0.2     // 감쇠 정도
    }

    // 파이프라인
    SubShader{
        Tags{ "Queue"="Transparent" "RenderType"="Transparent" }   // 투명 렌더 큐
        Cull Back
        ZWrite Off
        Blend SrcAlpha OneMinusSrcAlpha
        // AlphaToMask On   // MSAA 사용 시 엣지 톱니 줄이고 싶으면 활성화

        Pass{
            CGPROGRAM
            #pragma vertex vert         // 버텍스 함수 지정
            #pragma fragment frag       // 프래그먼트 함수 지정
            #include "UnityCG.cginc"

            // CPU에서 GPU로 들어오는 데이터
            struct appdata {
                float4 vertex : POSITION;   // 버텍스 위치
                float2 uv     : TEXCOORD0;  // UV 좌표
            };

            // 버텍스에서 프래그먼트로 넘기는 값
            struct v2f {
                float4 pos : SV_POSITION;   // 화면 좌표
                float2 uv  : TEXCOORD0;     // UV 좌표
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _Thresh, _Smooth, _Despill;

            // 버텍스
            v2f vert (appdata v){
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);  // 위치 변환
                o.uv  = TRANSFORM_TEX(v.uv, _MainTex);   // UV 계산
                return o;
            }

            // Rec.709 luma
            // 사람 눈에 보이는 밝기를 반영하여 실제 밝기 계산하는 함수
            inline float luminance(float3 c){ return dot(c, float3(0.2126, 0.7152, 0.0722)); }

            // 프래그먼트
            fixed4 frag (v2f i) : SV_Target{
                fixed4 col = tex2D(_MainTex, i.uv);  // UV로 텍스처에서 색상 추출

                float m = max(col.r, max(col.g, col.b));  // RGB 중 최댓값 계산
                float lum = luminance(col.rgb);
                
                // 알파 계산; 어두울수록 투명, 밝을수록 불투명
                float alpha = smoothstep(_Thresh, _Thresh + _Smooth, m);

                // 경계 색상 조정
                float3 neutral = lerp(col.rgb * (1.0 + _Despill), col.rgb, alpha);

                return float4(neutral, alpha);
            }
            ENDCG
        }
    }
}
