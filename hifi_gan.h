#include <stdio.h>
#include "hifi_gan_weights.h"
#include <immintrin.h>

void conv_resblock0(float * matrix_in, float * matrix_out, float conv_weight[64][64][3], float conv_bias[64], int dilation, int padding, int frame_nums);
void conv_resblock1(float * matrix_in, float * matrix_out, float conv_weight[64][64][7], float conv_bias[64], int dilation, int padding, int frame_nums);
void conv_resblock2(float * matrix_in, float * matrix_out, float conv_weight[64][64][11], float conv_bias[64], int dilation, int padding, int frame_nums);

void conv_resblock3(float * matrix_in, float * matrix_out, float conv_weight[32][32][3], float conv_bias[32], int dilation, int padding, int frame_nums);
void conv_resblock4(float * matrix_in, float * matrix_out, float conv_weight[32][32][7], float conv_bias[32], int dilation, int padding, int frame_nums);
void conv_resblock5(float * matrix_in, float * matrix_out, float conv_weight[32][32][11], float conv_bias[32], int dilation, int padding, int frame_nums);

void conv_resblock6(float * matrix_in, float * matrix_out, float conv_weight[16][16][3], float conv_bias[16], int dilation, int padding, int frame_nums);
void conv_resblock7(float * matrix_in, float * matrix_out, float conv_weight[16][16][7], float conv_bias[16], int dilation, int padding, int frame_nums);
void conv_resblock8(float * matrix_in, float * matrix_out, float conv_weight[16][16][11], float conv_bias[16], int dilation, int padding, int frame_nums);

void conv_resblock9(float * matrix_in, float * matrix_out, float conv_weight[8][8][3], float conv_bias[8], int dilation, int padding, int frame_nums);
void conv_resblock10(float * matrix_in, float * matrix_out, float conv_weight[8][8][7], float conv_bias[8], int dilation, int padding, int frame_nums);
void conv_resblock11(float * matrix_in, float * matrix_out, float conv_weight[8][8][11], float conv_bias[8], int dilation, int padding, int frame_nums);


void conv_pre(float mel_spec[][80], float * conv_pre_out, int frame_nums){
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
	mel_spec_padding[frame_nums+3][i] = 0;
	mel_spec_padding[frame_nums+4][i] = 0;
	mel_spec_padding[frame_nums+5][i] = 0;
    }


    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 128; j++){
            for(k = 0; k < 80; k++){
		conv_pre_out[i*128+j] += mel_spec_padding[i][k] * conv_pre_weight[j][k][0]
		       	            + mel_spec_padding[i+1][k] * conv_pre_weight[j][k][1] 
				    + mel_spec_padding[i+2][k] * conv_pre_weight[j][k][2] 
				    + mel_spec_padding[i+3][k] * conv_pre_weight[j][k][3] 
				    + mel_spec_padding[i+4][k] * conv_pre_weight[j][k][4] 
				    + mel_spec_padding[i+5][k] * conv_pre_weight[j][k][5] 
				    + mel_spec_padding[i+6][k] * conv_pre_weight[j][k][6];
		}
	     conv_pre_out[i*128+j] += conv_pre_bias[j];
	     }
        } 
}

void conv_post(float * resblock11_out, float * conv_post_out, int frame_nums){
    int i;

    for(i = 0; i < frame_nums * 8; i++){
        if(resblock11_out[i] < 0) resblock11_out[i] = resblock11_out[i] * 0.01;
	}

    for(i = 0; i < frame_nums-6; i++){
        conv_post_out[i] = resblock11_out[8*i] * conv_post_weight[0][0] +
                            resblock11_out[8*i+1] * conv_post_weight[1][0] + 
                            resblock11_out[8*i+2] * conv_post_weight[2][0] +
                            resblock11_out[8*i+3] * conv_post_weight[3][0] +
                            resblock11_out[8*i+4] * conv_post_weight[4][0] +
                            resblock11_out[8*i+5] * conv_post_weight[5][0] + 
                            resblock11_out[8*i+6] * conv_post_weight[6][0] + 
							resblock11_out[8*i+7] * conv_post_weight[7][0] +

                            resblock11_out[8*i+8] * conv_post_weight[0][1] +
                            resblock11_out[8*i+9] * conv_post_weight[1][1] + 
                            resblock11_out[8*i+10] * conv_post_weight[2][1] +
                            resblock11_out[8*i+11] * conv_post_weight[3][1] +
                            resblock11_out[8*i+12] * conv_post_weight[4][1] +
                            resblock11_out[8*i+13] * conv_post_weight[5][1] + 
                            resblock11_out[8*i+14] * conv_post_weight[6][1] + 
							resblock11_out[8*i+15] * conv_post_weight[7][1] +
 
                            resblock11_out[8*i+16] * conv_post_weight[0][2] +
                            resblock11_out[8*i+17] * conv_post_weight[1][2] + 
                            resblock11_out[8*i+18] * conv_post_weight[2][2] +
                            resblock11_out[8*i+19] * conv_post_weight[3][2] +
                            resblock11_out[8*i+20] * conv_post_weight[4][2] +
                            resblock11_out[8*i+21] * conv_post_weight[5][2] + 
                            resblock11_out[8*i+22] * conv_post_weight[6][2] + 
							resblock11_out[8*i+23] * conv_post_weight[7][2] +

                            resblock11_out[8*i+24] * conv_post_weight[0][3] +
                            resblock11_out[8*i+25] * conv_post_weight[1][3] + 
                            resblock11_out[8*i+26] * conv_post_weight[2][3] +
                            resblock11_out[8*i+27] * conv_post_weight[3][3] +
                            resblock11_out[8*i+28] * conv_post_weight[4][3] +
                            resblock11_out[8*i+29] * conv_post_weight[5][3] + 
                            resblock11_out[8*i+30] * conv_post_weight[6][3] + 
							resblock11_out[8*i+31] * conv_post_weight[7][3] +

                            resblock11_out[8*i+32] * conv_post_weight[0][4] +
                            resblock11_out[8*i+33] * conv_post_weight[1][4] + 
                            resblock11_out[8*i+34] * conv_post_weight[2][4] +
                            resblock11_out[8*i+35] * conv_post_weight[3][4] +
                            resblock11_out[8*i+36] * conv_post_weight[4][4] +
                            resblock11_out[8*i+37] * conv_post_weight[5][4] + 
                            resblock11_out[8*i+38] * conv_post_weight[6][4] + 
							resblock11_out[8*i+39] * conv_post_weight[7][4] +

                            resblock11_out[8*i+40] * conv_post_weight[0][5] +
                            resblock11_out[8*i+41] * conv_post_weight[1][5] + 
                            resblock11_out[8*i+42] * conv_post_weight[2][5] +
                            resblock11_out[8*i+43] * conv_post_weight[3][5] +
                            resblock11_out[8*i+44] * conv_post_weight[4][5] +
                            resblock11_out[8*i+45] * conv_post_weight[5][5] + 
                            resblock11_out[8*i+46] * conv_post_weight[6][5] + 
							resblock11_out[8*i+47] * conv_post_weight[7][5] +

                            resblock11_out[8*i+48] * conv_post_weight[0][6] +
                            resblock11_out[8*i+49] * conv_post_weight[1][6] + 
                            resblock11_out[8*i+50] * conv_post_weight[2][6] +
                            resblock11_out[8*i+51] * conv_post_weight[3][6] +
                            resblock11_out[8*i+52] * conv_post_weight[4][6] +
                            resblock11_out[8*i+53] * conv_post_weight[5][6] + 
                            resblock11_out[8*i+54] * conv_post_weight[6][6] + 
							resblock11_out[8*i+55] * conv_post_weight[7][6] + conv_post_bias;								
    }
    for(i = frame_nums-6; i < frame_nums; i++){
        conv_post_out[i] = 0;
    }
}

void ups0(float * conv_pre_out, float * ups0_out, int frame_nums){
    int i;
    int j;
    int k;
    int m;
    int n;
    
    float pre_sample[16];
    float pos_sample[16];
    
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

void ups1(float * resblock2_out , float * ups1_out, int frame_nums){
    int i;
    int j;
    int k;
    int m;
    int n;

    float pre_sample[16];
    float pos_sample[16];

    for(i = 0; i < frame_nums*64; i++){
        if (resblock2_out[i]<0) resblock2_out[i] *= 0.1;
    }

    float ups1_out_temp[frame_nums*8][32];

    for(i = 0; i < frame_nums - 1; i++){
        for(j = 0; j < 32; j++){
            for(k = 0; k < 16; k++){
                pre_sample[k] = 0.0;
                pos_sample[k] = 0.0;
            }

            for(k = 0; k < 16; k++){
                for(m = 0; m < 64; m++){
                    pre_sample[k] += resblock2_out[i*64+m] * ups_1_weight[m][j][k];
                    pos_sample[k] += resblock2_out[(i+1)*64+m] * ups_1_weight[m][j][k];
                }
            }

            if(i==0){
                ups1_out_temp[0][j] = pre_sample[4] + ups_1_bias[j];
                ups1_out_temp[1][j] = pre_sample[5] + ups_1_bias[j];
                ups1_out_temp[2][j] = pre_sample[6] + ups_1_bias[j];
                ups1_out_temp[3][j] = pre_sample[7] + ups_1_bias[j];
            }

            ups1_out_temp[8*i+4][j] = pre_sample[8] + pos_sample[0] + ups_1_bias[j];
            ups1_out_temp[8*i+5][j] = pre_sample[9] + pos_sample[1] + ups_1_bias[j];
            ups1_out_temp[8*i+6][j] = pre_sample[10] + pos_sample[2] + ups_1_bias[j];
            ups1_out_temp[8*i+7][j] = pre_sample[11] + pos_sample[3] + ups_1_bias[j];
            ups1_out_temp[8*i+8][j] = pre_sample[12] + pos_sample[4] + ups_1_bias[j];
            ups1_out_temp[8*i+9][j] = pre_sample[13] + pos_sample[5] + ups_1_bias[j];
            ups1_out_temp[8*i+10][j] = pre_sample[14] + pos_sample[6] + ups_1_bias[j];
            ups1_out_temp[8*i+11][j] = pre_sample[15] + pos_sample[7] + ups_1_bias[j];

            if(i==frame_nums-2){
                ups1_out_temp[8*i+12][j] = pos_sample[8] + ups_1_bias[j];
                ups1_out_temp[8*i+13][j] = pos_sample[9] + ups_1_bias[j];
                ups1_out_temp[8*i+14][j] = pos_sample[10] + ups_1_bias[j];
                ups1_out_temp[8*i+15][j] = pos_sample[11] + ups_1_bias[j];

            }

         }
    }

    for(i = 0; i < frame_nums*8; i++){
        for(j = 0; j < 32; j++){
            ups1_out[32*i+j] = ups1_out_temp[i][j];
        }
    }
}

void ups2(float * resblock5_out , float * ups2_out, int frame_nums){
    int i;
    int j;
    int k;
    int m;
    int n;

    float pre_sample[4];
    float pos_sample[4];

    for(i = 0; i < frame_nums * 32; i++){
        if (resblock5_out[i]<0) resblock5_out[i] *= 0.1;
    }

    float ups2_out_temp[frame_nums*2][16]; //double its frames after upsampling

    for(i = 0; i < frame_nums - 1; i++){
        for(j = 0; j < 16; j++){
            for(k = 0; k < 4; k++){
                pre_sample[k] = 0.0;
                pos_sample[k] = 0.0;
            }

            for(k = 0; k < 4; k++){
                for(m = 0; m < 32; m++){
                    pre_sample[k] += resblock5_out[i*32+m] * ups_2_weight[m][j][k];
                    pos_sample[k] += resblock5_out[(i+1)*32+m] * ups_2_weight[m][j][k];
                }
            }

            if(i==0){
                ups2_out_temp[0][j] = pre_sample[1] + ups_2_bias[j];
            }

            ups2_out_temp[2*i+1][j] = pre_sample[2] + pos_sample[0] + ups_2_bias[j];
            ups2_out_temp[2*i+2][j] = pre_sample[3] + pos_sample[1] + ups_2_bias[j];

            if(i==frame_nums-2){
                ups2_out_temp[2*i+3][j] = pos_sample[2] + ups_2_bias[j];

            }

         }
    }

    for(i = 0; i < frame_nums*2; i++){ //double its frames
        for(j = 0; j < 16; j++){ //channel==16
            ups2_out[16*i+j] = ups2_out_temp[i][j];
        }
    }
}


void ups3(float * resblock8_out , float * ups3_out, int frame_nums){
    int i;
    int j;
    int k;
    int m;
    int n;

    float pre_sample[4];
    float pos_sample[4];

    for(i = 0; i < frame_nums * 8; i++){
        if (resblock8_out[i]<0) resblock8_out[i] *= 0.1;
    }



    float ups3_out_temp[frame_nums*2][8]; //double its frames after upsampling

    for(i = 0; i < frame_nums - 1; i++){
        for(j = 0; j < 8; j++){
            for(k = 0; k < 4; k++){
                pre_sample[k] = 0.0;
                pos_sample[k] = 0.0;
            }

            for(k = 0; k < 4; k++){
                for(m = 0; m < 16; m++){
                    pre_sample[k] += resblock8_out[i*16+m] * ups_3_weight[m][j][k];
                    pos_sample[k] += resblock8_out[(i+1)*16+m] * ups_3_weight[m][j][k];
                }
            }

            if(i==0){
                ups3_out_temp[0][j] = pre_sample[1] + ups_3_bias[j];
            }

            ups3_out_temp[2*i+1][j] = pre_sample[2] + pos_sample[0] + ups_3_bias[j];
            ups3_out_temp[2*i+2][j] = pre_sample[3] + pos_sample[1] + ups_3_bias[j];

            if(i==frame_nums-2){
                ups3_out_temp[2*i+3][j] = pos_sample[2] + ups_3_bias[j];

            }

         }
    }

    for(i = 0; i < frame_nums*2; i++){ //double its frames
        for(j = 0; j < 8; j++){ //channel==8
            ups3_out[8*i+j] = ups3_out_temp[i][j];
        }
    }
}

                                 
void resblock0(float * ups0_out, float * resblock0_output, int frame_nums){
    int i;
    float * resblock0_conv10 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock0_conv11 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock0_conv12 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock0_conv20 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock0_conv21 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock0_conv22 = (float*) calloc(frame_nums * 64, sizeof(float));

    conv_resblock0(ups0_out, resblock0_conv10, resblocks_0_convs1_0_weight, resblocks_0_convs1_0_bias, 1, 1, frame_nums);
    conv_resblock0(resblock0_conv10, resblock0_conv20, resblocks_0_convs2_0_weight, resblocks_0_convs2_0_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 64; i++){
        resblock0_conv20[i] += ups0_out[i];
    }

    conv_resblock0(resblock0_conv20, resblock0_conv11, resblocks_0_convs1_1_weight, resblocks_0_convs1_1_bias, 3, 3, frame_nums);
    conv_resblock0(resblock0_conv11, resblock0_conv21, resblocks_0_convs2_1_weight, resblocks_0_convs2_1_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 64; i++){
        resblock0_conv21[i] += resblock0_conv20[i];
    }

    conv_resblock0(resblock0_conv21, resblock0_conv12, resblocks_0_convs1_2_weight, resblocks_0_convs1_2_bias, 5, 5, frame_nums);
    conv_resblock0(resblock0_conv12, resblock0_conv22, resblocks_0_convs2_2_weight, resblocks_0_convs2_2_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 64; i++){
        resblock0_conv22[i] += resblock0_conv21[i];
    }


    for(i = 0; i < frame_nums*64; i++){
        resblock0_output[i] = resblock0_conv22[i];
    }

    for(i = 0; i < 64; i++){
        //printf("%f,", resblock0_conv10[i]);
    }


    free(resblock0_conv10);
    resblock0_conv10 = NULL;

    free(resblock0_conv10);
    resblock0_conv11 = NULL;

    free(resblock0_conv10);
    resblock0_conv12 = NULL;

    free(resblock0_conv10);
    resblock0_conv20 = NULL;

    free(resblock0_conv10);
    resblock0_conv21 = NULL;

    free(resblock0_conv10);
    resblock0_conv22 = NULL;
}

void conv_resblock0(float * matrix_in, float * matrix_out, float conv_weight[64][64][3], float conv_bias[64], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][64];
    
    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 64; j++){
            matrix_in_padding[i+padding][j] = matrix_in[64*i+j];
	    if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
	}
    }

    for(i = 0; i < 64; i++){
	for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 64; j++){
            for(k = 0; k < 64; k++){
                matrix_out[i*64+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2];
            }
            matrix_out[i*64+j] += conv_bias[j];
        }
    }
}



void resblock1(float * ups0_out, float * resblock1_output, int frame_nums){
    int i;
    float * resblock1_conv10 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock1_conv11 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock1_conv12 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock1_conv20 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock1_conv21 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock1_conv22 = (float*) calloc(frame_nums * 64, sizeof(float));

    conv_resblock1(ups0_out, resblock1_conv10, resblocks_1_convs1_0_weight, resblocks_1_convs1_0_bias, 1, 3, frame_nums);
    conv_resblock1(resblock1_conv10, resblock1_conv20, resblocks_1_convs2_0_weight, resblocks_1_convs2_0_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 64; i++){
        resblock1_conv20[i] += ups0_out[i];
    }

    conv_resblock1(resblock1_conv20, resblock1_conv11, resblocks_1_convs1_1_weight, resblocks_1_convs1_1_bias, 3, 9, frame_nums);
    conv_resblock1(resblock1_conv11, resblock1_conv21, resblocks_1_convs2_1_weight, resblocks_1_convs2_1_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 64; i++){
        resblock1_conv21[i] += resblock1_conv20[i];
    }

    conv_resblock1(resblock1_conv21, resblock1_conv12, resblocks_1_convs1_2_weight, resblocks_1_convs1_2_bias, 5, 15, frame_nums);
    conv_resblock1(resblock1_conv12, resblock1_conv22, resblocks_1_convs2_2_weight, resblocks_1_convs2_2_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 64; i++){
        resblock1_conv22[i] += resblock1_conv21[i];
    }


    for(i = 0; i < frame_nums*64; i++){
        resblock1_output[i] = resblock1_conv22[i];
    }

    //for(i = 0; i < 64; i++){
    //    printf("%f,", resblock1_output[i]);
    //}
    
    free(resblock1_conv10);
    resblock1_conv10 = NULL;

    free(resblock1_conv11);
    resblock1_conv11 = NULL;

    free(resblock1_conv12);
    resblock1_conv12 = NULL;

    free(resblock1_conv20);
    resblock1_conv20 = NULL;

    free(resblock1_conv21);
    resblock1_conv21 = NULL;

    free(resblock1_conv22);
    resblock1_conv22 = NULL;
}


void conv_resblock1(float * matrix_in, float * matrix_out, float conv_weight[64][64][7], float conv_bias[64], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][64];

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 64; j++){
            matrix_in_padding[i+padding][j] = matrix_in[64*i+j];
            if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
        }
    }

    for(i = 0; i < 64; i++){
        for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 64; j++){
            for(k = 0; k < 64; k++){
                matrix_out[i*64+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2]
                    + matrix_in_padding[i+3*dilation][k] * conv_weight[j][k][3]
                    + matrix_in_padding[i+4*dilation][k] * conv_weight[j][k][4]
		    + matrix_in_padding[i+5*dilation][k] * conv_weight[j][k][5]
		    + matrix_in_padding[i+6*dilation][k] * conv_weight[j][k][6];
            }
            matrix_out[i*64+j] += conv_bias[j];
        }
    }
}


void resblock2(float * ups0_out, float * resblock2_output, int frame_nums){
    int i;
    float * resblock2_conv10 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock2_conv11 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock2_conv12 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock2_conv20 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock2_conv21 = (float*) calloc(frame_nums * 64, sizeof(float));
    float * resblock2_conv22 = (float*) calloc(frame_nums * 64, sizeof(float));

    conv_resblock2(ups0_out, resblock2_conv10, resblocks_2_convs1_0_weight, resblocks_2_convs1_0_bias, 1, 5, frame_nums);
    conv_resblock2(resblock2_conv10, resblock2_conv20, resblocks_2_convs2_0_weight, resblocks_2_convs2_0_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 64; i++){
        resblock2_conv20[i] += ups0_out[i];
    }

    conv_resblock2(resblock2_conv20, resblock2_conv11, resblocks_2_convs1_1_weight, resblocks_2_convs1_1_bias, 3, 15, frame_nums);
    conv_resblock2(resblock2_conv11, resblock2_conv21, resblocks_2_convs2_1_weight, resblocks_2_convs2_1_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 64; i++){
        resblock2_conv21[i] += resblock2_conv20[i];
    }

    conv_resblock2(resblock2_conv21, resblock2_conv12, resblocks_2_convs1_2_weight, resblocks_2_convs1_2_bias, 5, 25, frame_nums);
    conv_resblock2(resblock2_conv12, resblock2_conv22, resblocks_2_convs2_2_weight, resblocks_2_convs2_2_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 64; i++){
        resblock2_conv22[i] += resblock2_conv21[i];
    }


    for(i = 0; i < frame_nums*64; i++){
        resblock2_output[i] = resblock2_conv22[i];
    }
    for(i = 0; i < 64; i++){
    //    printf("%f,", resblock2_conv20[i]);
    }

    free(resblock2_conv10);
    resblock2_conv10 = NULL;

    free(resblock2_conv11);
    resblock2_conv11 = NULL;

    free(resblock2_conv12);
    resblock2_conv12 = NULL;

    free(resblock2_conv20);
    resblock2_conv20 = NULL;

    free(resblock2_conv21);
    resblock2_conv21 = NULL;

    free(resblock2_conv22);
    resblock2_conv22 = NULL;
}

void conv_resblock2(float * matrix_in, float * matrix_out, float conv_weight[64][64][11], float conv_bias[64], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][64];

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 64; j++){
            matrix_in_padding[i+padding][j] = matrix_in[64*i+j];
            if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
        }
    }

    for(i = 0; i < 64; i++){
        for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 64; j++){
            for(k = 0; k < 64; k++){
                matrix_out[i*64+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2]
                    + matrix_in_padding[i+3*dilation][k] * conv_weight[j][k][3]
                    + matrix_in_padding[i+4*dilation][k] * conv_weight[j][k][4]
                    + matrix_in_padding[i+5*dilation][k] * conv_weight[j][k][5]
                    + matrix_in_padding[i+6*dilation][k] * conv_weight[j][k][6]
		    + matrix_in_padding[i+7*dilation][k] * conv_weight[j][k][7]
		    + matrix_in_padding[i+8*dilation][k] * conv_weight[j][k][8]
		    + matrix_in_padding[i+9*dilation][k] * conv_weight[j][k][9]
		    + matrix_in_padding[i+10*dilation][k] * conv_weight[j][k][10];
            }
            matrix_out[i*64+j] += conv_bias[j];
        }
    }
}

void resblock3(float * ups1_out, float * resblock3_output, int frame_nums){
    int i;
    float * resblock3_conv10 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock3_conv11 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock3_conv12 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock3_conv20 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock3_conv21 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock3_conv22 = (float*) calloc(frame_nums * 32, sizeof(float));

    conv_resblock3(ups1_out, resblock3_conv10, resblocks_3_convs1_0_weight, resblocks_3_convs1_0_bias, 1, 1, frame_nums);
    conv_resblock3(resblock3_conv10, resblock3_conv20, resblocks_3_convs2_0_weight, resblocks_3_convs2_0_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 32; i++){
        resblock3_conv20[i] += ups1_out[i];
    }

    conv_resblock3(resblock3_conv20, resblock3_conv11, resblocks_3_convs1_1_weight, resblocks_3_convs1_1_bias, 3, 3, frame_nums);
    conv_resblock3(resblock3_conv11, resblock3_conv21, resblocks_3_convs2_1_weight, resblocks_3_convs2_1_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 32; i++){
        resblock3_conv21[i] += resblock3_conv20[i];
    }

    conv_resblock3(resblock3_conv21, resblock3_conv12, resblocks_3_convs1_2_weight, resblocks_3_convs1_2_bias, 5, 5, frame_nums);
    conv_resblock3(resblock3_conv12, resblock3_conv22, resblocks_3_convs2_2_weight, resblocks_3_convs2_2_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 32; i++){
        resblock3_conv22[i] += resblock3_conv21[i];
    }


    for(i = 0; i < frame_nums*32; i++){
        resblock3_output[i] = resblock3_conv22[i];
    }


    free(resblock3_conv10);
    resblock3_conv10 = NULL;

    free(resblock3_conv10);
    resblock3_conv11 = NULL;

    free(resblock3_conv10);
    resblock3_conv12 = NULL;

    free(resblock3_conv10);
    resblock3_conv20 = NULL;

    free(resblock3_conv10);
    resblock3_conv21 = NULL;

    free(resblock3_conv10);
    resblock3_conv22 = NULL;
}

void conv_resblock3(float * matrix_in, float * matrix_out, float conv_weight[32][32][3], float conv_bias[32], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][32];
    
    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 32; j++){
            matrix_in_padding[i+padding][j] = matrix_in[32*i+j];
	    if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
	}
    }

    for(i = 0; i < 32; i++){
	for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 32; j++){
            for(k = 0; k < 32; k++){
                matrix_out[i*32+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2];
            }
            matrix_out[i*32+j] += conv_bias[j];
        }
    }
}



void resblock4(float * ups1_out, float * resblock4_output, int frame_nums){
    int i;
    float * resblock4_conv10 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock4_conv11 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock4_conv12 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock4_conv20 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock4_conv21 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock4_conv22 = (float*) calloc(frame_nums * 32, sizeof(float));

    conv_resblock4(ups1_out, resblock4_conv10, resblocks_4_convs1_0_weight, resblocks_4_convs1_0_bias, 1, 3, frame_nums);
    conv_resblock4(resblock4_conv10, resblock4_conv20, resblocks_4_convs2_0_weight, resblocks_4_convs2_0_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 32; i++){
        resblock4_conv20[i] += ups1_out[i];
    }

    conv_resblock4(resblock4_conv20, resblock4_conv11, resblocks_4_convs1_1_weight, resblocks_4_convs1_1_bias, 3, 9, frame_nums);
    conv_resblock4(resblock4_conv11, resblock4_conv21, resblocks_4_convs2_1_weight, resblocks_4_convs2_1_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 32; i++){
        resblock4_conv21[i] += resblock4_conv20[i];
    }

    conv_resblock4(resblock4_conv21, resblock4_conv12, resblocks_4_convs1_2_weight, resblocks_4_convs1_2_bias, 5, 15, frame_nums);
    conv_resblock4(resblock4_conv12, resblock4_conv22, resblocks_4_convs2_2_weight, resblocks_4_convs2_2_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 32; i++){
        resblock4_conv22[i] += resblock4_conv21[i];
    }


    for(i = 0; i < frame_nums*32; i++){
        resblock4_output[i] = resblock4_conv22[i];
    }

    free(resblock4_conv10);
    resblock4_conv10 = NULL;

    free(resblock4_conv11);
    resblock4_conv11 = NULL;

    free(resblock4_conv12);
    resblock4_conv12 = NULL;

    free(resblock4_conv20);
    resblock4_conv20 = NULL;

    free(resblock4_conv21);
    resblock4_conv21 = NULL;

    free(resblock4_conv22);
    resblock4_conv22 = NULL;
}


void conv_resblock4(float * matrix_in, float * matrix_out, float conv_weight[32][32][7], float conv_bias[64], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][32];

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 32; j++){
            matrix_in_padding[i+padding][j] = matrix_in[32*i+j];
            if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
        }
    }

    for(i = 0; i < 32; i++){
        for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 32; j++){
            for(k = 0; k < 32; k++){
                matrix_out[i*32+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2]
                    + matrix_in_padding[i+3*dilation][k] * conv_weight[j][k][3]
                    + matrix_in_padding[i+4*dilation][k] * conv_weight[j][k][4]
		    + matrix_in_padding[i+5*dilation][k] * conv_weight[j][k][5]
		    + matrix_in_padding[i+6*dilation][k] * conv_weight[j][k][6];
            }
            matrix_out[i*32+j] += conv_bias[j];
        }
    }
}


void resblock5(float * ups1_out, float * resblock5_output, int frame_nums){
    int i;
    float * resblock5_conv10 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock5_conv11 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock5_conv12 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock5_conv20 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock5_conv21 = (float*) calloc(frame_nums * 32, sizeof(float));
    float * resblock5_conv22 = (float*) calloc(frame_nums * 32, sizeof(float));

    conv_resblock5(ups1_out, resblock5_conv10, resblocks_5_convs1_0_weight, resblocks_5_convs1_0_bias, 1, 5, frame_nums);
    conv_resblock5(resblock5_conv10, resblock5_conv20, resblocks_5_convs2_0_weight, resblocks_5_convs2_0_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 32; i++){
        resblock5_conv20[i] += ups1_out[i];
    }

    conv_resblock5(resblock5_conv20, resblock5_conv11, resblocks_5_convs1_1_weight, resblocks_5_convs1_1_bias, 3, 15, frame_nums);
    conv_resblock5(resblock5_conv11, resblock5_conv21, resblocks_5_convs2_1_weight, resblocks_5_convs2_1_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 32; i++){
        resblock5_conv21[i] += resblock5_conv20[i];
    }

    conv_resblock5(resblock5_conv21, resblock5_conv12, resblocks_5_convs1_2_weight, resblocks_5_convs1_2_bias, 5, 25, frame_nums);
    conv_resblock5(resblock5_conv12, resblock5_conv22, resblocks_5_convs2_2_weight, resblocks_5_convs2_2_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 32; i++){
        resblock5_conv22[i] += resblock5_conv21[i];
    }


    for(i = 0; i < frame_nums*32; i++){
        resblock5_output[i] = resblock5_conv22[i];
    }
    
    free(resblock5_conv10);
    resblock5_conv10 = NULL;

    free(resblock5_conv11);
    resblock5_conv11 = NULL;

    free(resblock5_conv12);
    resblock5_conv12 = NULL;

    free(resblock5_conv20);
    resblock5_conv20 = NULL;

    free(resblock5_conv21);
    resblock5_conv21 = NULL;

    free(resblock5_conv22);
    resblock5_conv22 = NULL;
}

void conv_resblock5(float * matrix_in, float * matrix_out, float conv_weight[32][32][11], float conv_bias[32], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][32];

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 32; j++){
            matrix_in_padding[i+padding][j] = matrix_in[32*i+j];
            if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
        }
    }

    for(i = 0; i < 32; i++){
        for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 32; j++){
            for(k = 0; k < 32; k++){
                matrix_out[i*32+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2]
                    + matrix_in_padding[i+3*dilation][k] * conv_weight[j][k][3]
                    + matrix_in_padding[i+4*dilation][k] * conv_weight[j][k][4]
                    + matrix_in_padding[i+5*dilation][k] * conv_weight[j][k][5]
                    + matrix_in_padding[i+6*dilation][k] * conv_weight[j][k][6]
		    + matrix_in_padding[i+7*dilation][k] * conv_weight[j][k][7]
		    + matrix_in_padding[i+8*dilation][k] * conv_weight[j][k][8]
		    + matrix_in_padding[i+9*dilation][k] * conv_weight[j][k][9]
		    + matrix_in_padding[i+10*dilation][k] * conv_weight[j][k][10];
            }
            matrix_out[i*32+j] += conv_bias[j];
        }
    }
}


void resblock6(float * ups2_out, float * resblock6_output, int frame_nums){
    int i;
    float * resblock6_conv10 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock6_conv11 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock6_conv12 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock6_conv20 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock6_conv21 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock6_conv22 = (float*) calloc(frame_nums * 16, sizeof(float));

    conv_resblock6(ups2_out, resblock6_conv10, resblocks_6_convs1_0_weight, resblocks_6_convs1_0_bias, 1, 1, frame_nums);
    conv_resblock6(resblock6_conv10, resblock6_conv20, resblocks_6_convs2_0_weight, resblocks_6_convs2_0_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 16; i++){
        resblock6_conv20[i] += ups2_out[i];
    }

    conv_resblock6(resblock6_conv20, resblock6_conv11, resblocks_6_convs1_1_weight, resblocks_6_convs1_1_bias, 3, 3, frame_nums);
    conv_resblock6(resblock6_conv11, resblock6_conv21, resblocks_6_convs2_1_weight, resblocks_6_convs2_1_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 16; i++){
        resblock6_conv21[i] += resblock6_conv20[i];
    }

    conv_resblock6(resblock6_conv21, resblock6_conv12, resblocks_6_convs1_2_weight, resblocks_6_convs1_2_bias, 5, 5, frame_nums);
    conv_resblock6(resblock6_conv12, resblock6_conv22, resblocks_6_convs2_2_weight, resblocks_6_convs2_2_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 16; i++){
        resblock6_conv22[i] += resblock6_conv21[i];
    }


    for(i = 0; i < frame_nums * 16; i++){
        resblock6_output[i] = resblock6_conv22[i];
    }


    free(resblock6_conv10);
    resblock6_conv10 = NULL;

    free(resblock6_conv10);
    resblock6_conv11 = NULL;

    free(resblock6_conv10);
    resblock6_conv12 = NULL;

    free(resblock6_conv10);
    resblock6_conv20 = NULL;

    free(resblock6_conv10);
    resblock6_conv21 = NULL;

    free(resblock6_conv10);
    resblock6_conv22 = NULL;
}

void conv_resblock6(float * matrix_in, float * matrix_out, float conv_weight[16][16][3], float conv_bias[16], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][16];
    
    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 16; j++){
            matrix_in_padding[i+padding][j] = matrix_in[16*i+j];
	    if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
	}
    }

    for(i = 0; i < 16; i++){
	for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 16; j++){
            for(k = 0; k < 16; k++){
                matrix_out[i*16+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2];
            }
            matrix_out[i*16+j] += conv_bias[j];
        }
    }
}



void resblock7(float * ups2_out, float * resblock7_output, int frame_nums){
    int i;
    float * resblock7_conv10 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock7_conv11 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock7_conv12 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock7_conv20 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock7_conv21 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock7_conv22 = (float*) calloc(frame_nums * 16, sizeof(float));

    conv_resblock7(ups2_out, resblock7_conv10, resblocks_7_convs1_0_weight, resblocks_7_convs1_0_bias, 1, 3, frame_nums);
    conv_resblock7(resblock7_conv10, resblock7_conv20, resblocks_7_convs2_0_weight, resblocks_7_convs2_0_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 16; i++){
        resblock7_conv20[i] += ups2_out[i];
    }

    conv_resblock7(resblock7_conv20, resblock7_conv11, resblocks_7_convs1_1_weight, resblocks_7_convs1_1_bias, 3, 9, frame_nums);
    conv_resblock7(resblock7_conv11, resblock7_conv21, resblocks_7_convs2_1_weight, resblocks_7_convs2_1_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 16; i++){
        resblock7_conv21[i] += resblock7_conv20[i];
    }

    conv_resblock7(resblock7_conv21, resblock7_conv12, resblocks_7_convs1_2_weight, resblocks_7_convs1_2_bias, 5, 15, frame_nums);
    conv_resblock7(resblock7_conv12, resblock7_conv22, resblocks_7_convs2_2_weight, resblocks_7_convs2_2_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 16; i++){
        resblock7_conv22[i] += resblock7_conv21[i];
    }


    for(i = 0; i < frame_nums*16; i++){
        resblock7_output[i] = resblock7_conv22[i];
    }

    free(resblock7_conv10);
    resblock7_conv10 = NULL;

    free(resblock7_conv11);
    resblock7_conv11 = NULL;

    free(resblock7_conv12);
    resblock7_conv12 = NULL;

    free(resblock7_conv20);
    resblock7_conv20 = NULL;

    free(resblock7_conv21);
    resblock7_conv21 = NULL;

    free(resblock7_conv22);
    resblock7_conv22 = NULL;
}


void conv_resblock7(float * matrix_in, float * matrix_out, float conv_weight[16][16][7], float conv_bias[64], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][16];

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 16; j++){
            matrix_in_padding[i+padding][j] = matrix_in[16*i+j];
            if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
        }
    }

    for(i = 0; i < 16; i++){
        for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 16; j++){
            for(k = 0; k < 16; k++){
                matrix_out[i*16+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2]
                    + matrix_in_padding[i+3*dilation][k] * conv_weight[j][k][3]
                    + matrix_in_padding[i+4*dilation][k] * conv_weight[j][k][4]
		    + matrix_in_padding[i+5*dilation][k] * conv_weight[j][k][5]
		    + matrix_in_padding[i+6*dilation][k] * conv_weight[j][k][6];
            }
            matrix_out[i*16+j] += conv_bias[j];
        }
    }
}


void resblock8(float * ups2_out, float * resblock8_output, int frame_nums){
    int i;
    float * resblock8_conv10 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock8_conv11 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock8_conv12 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock8_conv20 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock8_conv21 = (float*) calloc(frame_nums * 16, sizeof(float));
    float * resblock8_conv22 = (float*) calloc(frame_nums * 16, sizeof(float));

    conv_resblock8(ups2_out, resblock8_conv10, resblocks_8_convs1_0_weight, resblocks_8_convs1_0_bias, 1, 5, frame_nums);
    conv_resblock8(resblock8_conv10, resblock8_conv20, resblocks_8_convs2_0_weight, resblocks_8_convs2_0_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 16; i++){
        resblock8_conv20[i] += ups2_out[i];
    }

    conv_resblock8(resblock8_conv20, resblock8_conv11, resblocks_8_convs1_1_weight, resblocks_8_convs1_1_bias, 3, 15, frame_nums);
    conv_resblock8(resblock8_conv11, resblock8_conv21, resblocks_8_convs2_1_weight, resblocks_8_convs2_1_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 16; i++){
        resblock8_conv21[i] += resblock8_conv20[i];
    }

    conv_resblock8(resblock8_conv21, resblock8_conv12, resblocks_8_convs1_2_weight, resblocks_8_convs1_2_bias, 5, 25, frame_nums);
    conv_resblock8(resblock8_conv12, resblock8_conv22, resblocks_8_convs2_2_weight, resblocks_8_convs2_2_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 16; i++){
        resblock8_conv22[i] += resblock8_conv21[i];
    }


    for(i = 0; i < frame_nums*16; i++){
        resblock8_output[i] = resblock8_conv22[i];
    }
    
    free(resblock8_conv10);
    resblock8_conv10 = NULL;

    free(resblock8_conv11);
    resblock8_conv11 = NULL;

    free(resblock8_conv12);
    resblock8_conv12 = NULL;

    free(resblock8_conv20);
    resblock8_conv20 = NULL;

    free(resblock8_conv21);
    resblock8_conv21 = NULL;

    free(resblock8_conv22);
    resblock8_conv22 = NULL;
}

void conv_resblock8(float * matrix_in, float * matrix_out, float conv_weight[16][16][11], float conv_bias[16], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][16];

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 16; j++){
            matrix_in_padding[i+padding][j] = matrix_in[16*i+j];
            if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
        }
    }

    for(i = 0; i < 16; i++){
        for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 16; j++){
            for(k = 0; k < 16; k++){
                matrix_out[i*16+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2]
                    + matrix_in_padding[i+3*dilation][k] * conv_weight[j][k][3]
                    + matrix_in_padding[i+4*dilation][k] * conv_weight[j][k][4]
                    + matrix_in_padding[i+5*dilation][k] * conv_weight[j][k][5]
                    + matrix_in_padding[i+6*dilation][k] * conv_weight[j][k][6]
		    + matrix_in_padding[i+7*dilation][k] * conv_weight[j][k][7]
		    + matrix_in_padding[i+8*dilation][k] * conv_weight[j][k][8]
		    + matrix_in_padding[i+9*dilation][k] * conv_weight[j][k][9]
		    + matrix_in_padding[i+10*dilation][k] * conv_weight[j][k][10];
            }
            matrix_out[i*16+j] += conv_bias[j];
        }
    }
}

void resblock9(float * ups3_out, float * resblock9_output, int frame_nums){
    int i;
    float * resblock9_conv10 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock9_conv11 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock9_conv12 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock9_conv20 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock9_conv21 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock9_conv22 = (float*) calloc(frame_nums * 8, sizeof(float));

    conv_resblock9(ups3_out, resblock9_conv10, resblocks_9_convs1_0_weight, resblocks_9_convs1_0_bias, 1, 1, frame_nums);
    conv_resblock9(resblock9_conv10, resblock9_conv20, resblocks_9_convs2_0_weight, resblocks_9_convs2_0_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 8; i++){
        resblock9_conv20[i] += ups3_out[i];
    }
    

    conv_resblock9(resblock9_conv20, resblock9_conv11, resblocks_9_convs1_1_weight, resblocks_9_convs1_1_bias, 3, 3, frame_nums);
    conv_resblock9(resblock9_conv11, resblock9_conv21, resblocks_9_convs2_1_weight, resblocks_9_convs2_1_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 8; i++){
        resblock9_conv21[i] += resblock9_conv20[i];
    }

    conv_resblock9(resblock9_conv21, resblock9_conv12, resblocks_9_convs1_2_weight, resblocks_9_convs1_2_bias, 5, 5, frame_nums);
    conv_resblock9(resblock9_conv12, resblock9_conv22, resblocks_9_convs2_2_weight, resblocks_9_convs2_2_bias, 1, 1, frame_nums);
    for(i = 0; i < frame_nums * 8; i++){
        resblock9_conv22[i] += resblock9_conv21[i];
    }


    for(i = 0; i < frame_nums*8; i++){
        resblock9_output[i] = resblock9_conv22[i];
    }


    free(resblock9_conv10);
    resblock9_conv10 = NULL;

    free(resblock9_conv10);
    resblock9_conv11 = NULL;

    free(resblock9_conv10);
    resblock9_conv12 = NULL;

    free(resblock9_conv10);
    resblock9_conv20 = NULL;

    free(resblock9_conv10);
    resblock9_conv21 = NULL;

    free(resblock9_conv10);
    resblock9_conv22 = NULL;
}

void conv_resblock9(float * matrix_in, float * matrix_out, float conv_weight[8][8][3], float conv_bias[8], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][8];
    
    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 8; j++){
            matrix_in_padding[i+padding][j] = matrix_in[8*i+j];
	    if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
	}
    }

    for(i = 0; i < 8; i++){
	for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 8; j++){
            for(k = 0; k < 8; k++){
                matrix_out[i*8+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2];
            }
            matrix_out[i*8+j] += conv_bias[j];
        }
    }
}



void resblock10(float * ups3_out, float * resblock10_output, int frame_nums){
    int i;
    float * resblock10_conv10 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock10_conv11 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock10_conv12 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock10_conv20 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock10_conv21 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock10_conv22 = (float*) calloc(frame_nums * 8, sizeof(float));

    conv_resblock10(ups3_out, resblock10_conv10, resblocks_10_convs1_0_weight, resblocks_10_convs1_0_bias, 1, 3, frame_nums);
    conv_resblock10(resblock10_conv10, resblock10_conv20, resblocks_10_convs2_0_weight, resblocks_10_convs2_0_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 8; i++){
        resblock10_conv20[i] += ups3_out[i];
    }

    conv_resblock10(resblock10_conv20, resblock10_conv11, resblocks_10_convs1_1_weight, resblocks_10_convs1_1_bias, 3, 9, frame_nums);
    conv_resblock10(resblock10_conv11, resblock10_conv21, resblocks_10_convs2_1_weight, resblocks_10_convs2_1_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 8; i++){
        resblock10_conv21[i] += resblock10_conv20[i];
    }

    conv_resblock10(resblock10_conv21, resblock10_conv12, resblocks_10_convs1_2_weight, resblocks_10_convs1_2_bias, 5, 15, frame_nums);
    conv_resblock10(resblock10_conv12, resblock10_conv22, resblocks_10_convs2_2_weight, resblocks_10_convs2_2_bias, 1, 3, frame_nums);
    for(i = 0; i < frame_nums * 8; i++){
        resblock10_conv22[i] += resblock10_conv21[i];
    }


    for(i = 0; i < frame_nums*8; i++){
        resblock10_output[i] = resblock10_conv22[i];
    }

    free(resblock10_conv10);
    resblock10_conv10 = NULL;

    free(resblock10_conv11);
    resblock10_conv11 = NULL;

    free(resblock10_conv12);
    resblock10_conv12 = NULL;

    free(resblock10_conv20);
    resblock10_conv20 = NULL;

    free(resblock10_conv21);
    resblock10_conv21 = NULL;

    free(resblock10_conv22);
    resblock10_conv22 = NULL;
}


void conv_resblock10(float * matrix_in, float * matrix_out, float conv_weight[8][8][7], float conv_bias[8], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][8];

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 8; j++){
            matrix_in_padding[i+padding][j] = matrix_in[8*i+j];
            if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
        }
    }

    for(i = 0; i < 8; i++){
        for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 8; j++){
            for(k = 0; k < 8; k++){
                matrix_out[i*8+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2]
                    + matrix_in_padding[i+3*dilation][k] * conv_weight[j][k][3]
                    + matrix_in_padding[i+4*dilation][k] * conv_weight[j][k][4]
		    + matrix_in_padding[i+5*dilation][k] * conv_weight[j][k][5]
		    + matrix_in_padding[i+6*dilation][k] * conv_weight[j][k][6];
            }
            matrix_out[i*8+j] += conv_bias[j];
        }
    }
}


void resblock11(float * ups3_out, float * resblock11_output, int frame_nums){
    int i;
    float * resblock11_conv10 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock11_conv11 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock11_conv12 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock11_conv20 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock11_conv21 = (float*) calloc(frame_nums * 8, sizeof(float));
    float * resblock11_conv22 = (float*) calloc(frame_nums * 8, sizeof(float));

    conv_resblock11(ups3_out, resblock11_conv10, resblocks_11_convs1_0_weight, resblocks_11_convs1_0_bias, 1, 5, frame_nums);
    conv_resblock11(resblock11_conv10, resblock11_conv20, resblocks_11_convs2_0_weight, resblocks_11_convs2_0_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 8; i++){
        resblock11_conv20[i] += ups3_out[i];
    }

    conv_resblock11(resblock11_conv20, resblock11_conv11, resblocks_11_convs1_1_weight, resblocks_11_convs1_1_bias, 3, 15, frame_nums);
    conv_resblock11(resblock11_conv11, resblock11_conv21, resblocks_11_convs2_1_weight, resblocks_11_convs2_1_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 8; i++){
        resblock11_conv21[i] += resblock11_conv20[i];
    }

    conv_resblock11(resblock11_conv21, resblock11_conv12, resblocks_11_convs1_2_weight, resblocks_11_convs1_2_bias, 5, 25, frame_nums);
    conv_resblock11(resblock11_conv12, resblock11_conv22, resblocks_11_convs2_2_weight, resblocks_11_convs2_2_bias, 1, 5, frame_nums);
    for(i = 0; i < frame_nums * 8; i++){
        resblock11_conv22[i] += resblock11_conv21[i];
    }


    for(i = 0; i < frame_nums * 8; i++){
        resblock11_output[i] = resblock11_conv22[i];
    }
    
    free(resblock11_conv10);
    resblock11_conv10 = NULL;

    free(resblock11_conv11);
    resblock11_conv11 = NULL;

    free(resblock11_conv12);
    resblock11_conv12 = NULL;

    free(resblock11_conv20);
    resblock11_conv20 = NULL;

    free(resblock11_conv21);
    resblock11_conv21 = NULL;

    free(resblock11_conv22);
    resblock11_conv22 = NULL;
}

void conv_resblock11(float * matrix_in, float * matrix_out, float conv_weight[8][8][11], float conv_bias[8], int dilation, int padding, int frame_nums){
    int i;
    int j;
    int k;

    float matrix_in_padding[frame_nums+2*padding][8];

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 8; j++){
            matrix_in_padding[i+padding][j] = matrix_in[8*i+j];
            if(matrix_in_padding[i+padding][j]<0) matrix_in_padding[i+padding][j] *= 0.1;
        }
    }

    for(i = 0; i < 8; i++){
        for(j = 0; j < padding; j++){
            matrix_in_padding[j][i] = 0;
            matrix_in_padding[frame_nums+j+padding][i] = 0;
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 8; j++){
            for(k = 0; k < 8; k++){
                matrix_out[i*8+j] += matrix_in_padding[i][k] * conv_weight[j][k][0]
                    + matrix_in_padding[i+dilation][k] * conv_weight[j][k][1]
                    + matrix_in_padding[i+2*dilation][k] * conv_weight[j][k][2]
                    + matrix_in_padding[i+3*dilation][k] * conv_weight[j][k][3]
                    + matrix_in_padding[i+4*dilation][k] * conv_weight[j][k][4]
                    + matrix_in_padding[i+5*dilation][k] * conv_weight[j][k][5]
                    + matrix_in_padding[i+6*dilation][k] * conv_weight[j][k][6]
		    + matrix_in_padding[i+7*dilation][k] * conv_weight[j][k][7]
		    + matrix_in_padding[i+8*dilation][k] * conv_weight[j][k][8]
		    + matrix_in_padding[i+9*dilation][k] * conv_weight[j][k][9]
		    + matrix_in_padding[i+10*dilation][k] * conv_weight[j][k][10];
            }
            matrix_out[i*8+j] += conv_bias[j];
        }
    }
}



