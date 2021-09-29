#include <stdio.h>


void conv_pre(float mel_spec[][80], float * conv_pre_out, float conv_weight[128][80][7], float conv_bias[128], int frame_nums){
    int i;
    int j;
    int k;
    float mel_spec_padding[frame_nums+6][80];
    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 80; j++){
            mel_spec_padding[i+3][j] = mel_spec[i][j];
        }
    }
    
    for(i = 0; i < 80; i++){
	mel_spec_padding[0][i] = 0;
	mel_spec_padding[1][i] = 0;
	mel_spec_padding[2][i] = 0;
	mel_spec_padding[248][i] = 0;
	mel_spec_padding[249][i] = 0;
	mel_spec_padding[250][i] = 0;
    }


    for(i = 0; i < 245; i++){
        for(j = 0; j < 128; j++){
            for(k = 0; k < 80; k++){
		conv_pre_out[i*128+j] += mel_spec_padding[i][k] * conv_weight[j][k][0]
		       	            + mel_spec_padding[i+1][k] * conv_weight[j][k][1] 
				    + mel_spec_padding[i+2][k] * conv_weight[j][k][2] 
				    + mel_spec_padding[i+3][k] * conv_weight[j][k][3] 
				    + mel_spec_padding[i+4][k] * conv_weight[j][k][4] 
				    + mel_spec_padding[i+5][k] * conv_weight[j][k][5] 
				    + mel_spec_padding[i+6][k] * conv_weight[j][k][6];
		}
	     conv_pre_out[i*128+j] += conv_bias[j];
	     }
        }    
    }
}



void ups0(float * conv_pre_out, float * ups0_out, float ups_0_weight[128][64][16], float ups_0_bias[64], int frame_nums){
    int i;
    int j;
    int k;
    int m;
    int n;
    
    float pre_sample[16];
    float pos_sample[16];
    
    for(i = 0; i < 245; i++){
        //printf("%f,", (conv_pre_out[i][128]));
    }


    
    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 128; j++){
            if (conv_pre_out[i*128+j]<0) conv_pre_out[i*128+j] *= 0.1;
        }
    }

    float ups0_out_temp[frame_nums*8][64];
   
    for(i = 0; i < frame_nums - 1; i++){
	for(j = 0; j < 64; j++){
	    for(k = 0; k < 16; k++){
                pre_sample[k] = 0.0;
                pos_sample[k] = 0.0;
            }

	    for(k = 0; k < 16; k++){
                for(m = 0; m < 128; m++){
		    pre_sample[k] += conv_pre_out[i*128+m] * ups_0_weight[m][j][k];
                    pos_sample[k] += conv_pre_out[(i+1)*128+m] * ups_0_weight[m][j][k];
                }
            }
            
	    if(i==0){
	        ups0_out_temp[0][j] = pre_sample[4] + ups_0_bias[j];
		ups0_out_temp[1][j] = pre_sample[5] + ups_0_bias[j];
		ups0_out_temp[2][j] = pre_sample[6] + ups_0_bias[j];
		ups0_out_temp[3][j] = pre_sample[7] + ups_0_bias[j];
	    }
            
            ups0_out_temp[8*i+4][j] = pre_sample[8] + pos_sample[0] + ups_0_bias[j];
	    ups0_out_temp[8*i+5][j] = pre_sample[9] + pos_sample[1] + ups_0_bias[j];
	    ups0_out_temp[8*i+6][j] = pre_sample[10] + pos_sample[2] + ups_0_bias[j];
	    ups0_out_temp[8*i+7][j] = pre_sample[11] + pos_sample[3] + ups_0_bias[j];
	    ups0_out_temp[8*i+8][j] = pre_sample[12] + pos_sample[4] + ups_0_bias[j];
	    ups0_out_temp[8*i+9][j] = pre_sample[13] + pos_sample[5] + ups_0_bias[j];
	    ups0_out_temp[8*i+10][j] = pre_sample[14] + pos_sample[6] + ups_0_bias[j];
	    ups0_out_temp[8*i+11][j] = pre_sample[15] + pos_sample[7] + ups_0_bias[j];
            
	    if(i==frame_nums-2){
	        ups0_out_temp[8*i+12][j] = pos_sample[8] + ups_0_bias[j];
                ups0_out_temp[8*i+13][j] = pos_sample[9] + ups_0_bias[j];
                ups0_out_temp[8*i+14][j] = pos_sample[10] + ups_0_bias[j];
                ups0_out_temp[8*i+15][j] = pos_sample[11] + ups_0_bias[j];

	    }
      
         }
    }
 
    for(i = 0; i < frame_nums*8; i++){
        for(j = 0; j < 64; j++){
	        ups0_out[64*i+j] = ups0_out_temp[i][j];
        }
    }
}



